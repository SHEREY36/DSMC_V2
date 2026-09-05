"""Build the versioned BL-compatible variational closure artifact."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from dsmc_v2_contracts import DIAGNOSTIC_NAMES, FEATURE_NAMES

from .estimate import estimate_node
from .fit_coefficients import fit_lambda1_coefficients
from .projections import (
    _legendre_nodes,
    _bridge_spline,
    angular_quantiles,
    conditional_energy_mean_map,
    energy_quantile_table,
    incoming_partition_density,
)


SCHEMA_VERSION = "2.3.0"
# Nodes on the a-axis of the energy sampler. a enters the kernel only
# linearly inside a smooth exponential tilt, so the quantile surface is
# gentle in a and 65 nodes interpolate it to ~1e-5 on the moments.
ENERGY_A_NODES = 65
ARTIFACT_TYPE = "bl_variational_closure"


def _node_key(values) -> tuple[float, float, float, int]:
    values = values if isinstance(values, dict) else values.metadata
    return (float(values["alpha"]), float(values["theta"]),
            float(values["aspect_ratio"]), int(values.get("ensemble_id", 0)))


def _path_key(path) -> tuple[float, float, float, int]:
    payload = json.loads((Path(path) / "metadata_v2.json").read_text())
    return _node_key(payload)


def _load_node_estimates(directory, expected_groups) -> list[dict]:
    expected_groups = {
        (key if len(key) == 4 else (*key, 0)): value
        for key, value in expected_groups.items()
    }
    nodes = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("alpha_*.json"))]
    keys = {_node_key(node) for node in nodes}
    if len(nodes) != len(keys) or keys != set(expected_groups):
        missing, extra = sorted(set(expected_groups) - keys), sorted(keys - set(expected_groups))
        raise ValueError(f"precomputed node grid mismatch; missing={missing}, extra={extra}")
    for node in nodes:
        expected = {Path(path).resolve() for path in expected_groups[_node_key(node)]}
        actual = {Path(path).resolve() for path in node.get("source_runs", [])}
        if actual != expected:
            raise ValueError(f"stale node estimate for {_node_key(node)}")
        if not node.get("qa", {}).get("precision_pass", node.get("qa", {}).get("sentinel_pass", False)):
            raise ValueError(f"node {_node_key(node)} has not passed closure QA")
    return sorted(nodes, key=_node_key)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _incoming_partition_mean(theta: float, order: int = 48) -> float:
    """Collision-pool mean for two independent Gamma(2) modal energies."""
    node, weight = np.polynomial.laguerre.laggauss(order)
    # Gamma(2,1) density is x*exp(-x); Laguerre supplies exp(-x).
    xx, yy = np.meshgrid(node, node, indexing="ij")
    ww = weight[:, None] * weight[None, :] * xx * yy
    return float(np.sum(ww * theta * xx / (theta * xx + yy)) / np.sum(ww))


def _energy_weighted_partition(theta: float) -> float:
    """Incoming collision-pool energy fraction relevant to temperature drift."""
    return float(theta / (1.0 + theta))


def _stability_rows(baseline: list[dict], bl=None) -> list[dict]:
    grouped = defaultdict(list)
    for node in baseline:
        grouped[(node["alpha"], node["aspect_ratio"])].append(node)
    rows = []
    for (alpha, ar), nodes in sorted(grouped.items()):
        nodes.sort(key=lambda row: row["theta"])
        if len(nodes) < 3:
            continue
        theta = np.array([row["theta"] for row in nodes])
        lambda1 = PchipInterpolator(theta, [row["energy"]["lambda1"] for row in nodes])
        lambda2 = PchipInterpolator(theta, [row["energy"]["lambda2"] for row in nodes])
        lambda3 = PchipInterpolator(theta, [row["energy"]["lambda3"] for row in nodes])
        lambda4 = PchipInterpolator(theta, [row["energy"].get("lambda4", 0.0)
                                            for row in nodes])
        partition_se = [row.get("uncertainty", {}).get(
            "mean_partition_out", {}).get("standard_error", np.nan) for row in nodes]
        mu_se = (PchipInterpolator(theta, partition_se)
                 if np.all(np.isfinite(partition_se))
                 else (lambda value: np.nan))
        grid, quad = _legendre_nodes(192, 0.0, 1.0)

        if alpha >= 1.0:
            mean_loss = 0.0
        elif bl is None:
            raise ValueError("the complete stability gate requires the frozen BL loss model")
        else:
            mean_loss = float(bl.parameters(alpha, ar)["mean_loss_fraction"])

        def post_collision_partition(value):
            """E[z_out] of the fitted kernel at the DSMC's own fractional loss.

            The incoming partition is averaged over its exact collision-weighted
            law, not evaluated at the energy-weighted fraction: those are two
            different objects and only the second belongs in the balance below.
            """
            mass = incoming_partition_density(value, grid) * quad
            mass = mass / np.sum(mass)
            parameters = np.array([float(lambda1(value)), float(lambda2(value)),
                                   float(lambda3(value))])
            mean_map = conditional_energy_mean_map(
                parameters, grid, offset=float(lambda4(value)) * mean_loss)
            return float(mass @ mean_map)

        def drift(value):
            incoming = _energy_weighted_partition(value)
            post_partition = post_collision_partition(value)
            delta_tr = (1.0 - mean_loss) * post_partition - incoming
            delta_rot = ((1.0 - mean_loss) * (1.0 - post_partition)
                         - (1.0 - incoming))
            return float((2.0 / 3.0) * delta_tr - value * delta_rot)

        dense = np.linspace(theta[0], theta[-1], 401)
        roots = []
        for left, right in zip(dense[:-1], dense[1:]):
            if drift(left) == 0.0:
                roots.append(float(left))
            elif drift(left) * drift(right) < 0.0:
                roots.append(float(brentq(drift, left, right)))
        roots = sorted({round(root, 12) for root in roots})
        derivative = None
        root_standard_error = None
        uncertainty_margin = None
        stable = False
        if len(roots) == 1:
            root = roots[0]
            step = max(1.0e-5, 1.0e-4 * root)
            derivative = (drift(min(theta[-1], root + step))
                          - drift(max(theta[0], root - step))) \
                / (min(theta[-1], root + step) - max(theta[0], root - step))
            stable = derivative < 0.0
            if np.isfinite(mu_se(root)) and derivative != 0.0:
                # First-order propagation through the post-collision partition,
                # whose bootstrap standard error stands in for the joint
                # uncertainty of the natural parameters.
                response = (1.0 - mean_loss) * (2.0 / 3.0 + root)
                drift_se = float(response * mu_se(root))
                root_standard_error = float(drift_se / abs(derivative))
                uncertainty_margin = float(
                    min(root - theta[0], theta[-1] - root) - 1.96 * root_standard_error)
        rows.append({"alpha": alpha, "aspect_ratio": ar, "roots": roots,
                     "unique_stable": bool(len(roots) == 1 and stable
                                           and uncertainty_margin is not None
                                           and uncertainty_margin > 0.0),
                     "drift_derivative": derivative,
                     "root_standard_error": root_standard_error,
                     "uncertainty_margin_to_hull": uncertainty_margin,
                     "mean_scalar_loss": mean_loss,
                     "drift_model": "complete_variational_partition_plus_frozen_BL_loss",
                     "includes_surface_derivatives": True})
    return rows


def build_artifact(run_directories, output_directory, bl=None,
                   n_bootstrap: int = 200, node_estimates=None) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in run_directories]
    if not paths:
        raise ValueError("no CTC runs supplied")
    grouped = defaultdict(list)
    for path in paths:
        grouped[_path_key(path)].append(path)
    if node_estimates:
        nodes = _load_node_estimates(node_estimates, grouped)
    else:
        nodes = [estimate_node(shards, bl, n_bootstrap=n_bootstrap)
                 for _, shards in sorted(grouped.items())]
        failed = [node for node in nodes if not node["qa"]["sentinel_pass"]]
        if failed:
            raise ValueError(f"{len(failed)} node(s) failed variational closure gates")
    baseline = [node for node in nodes if int(node["ensemble_id"]) == 0]
    if not baseline:
        raise ValueError("artifact requires baseline ensemble_id=0 nodes")
    baseline.sort(key=lambda row: (row["alpha"], row["theta"], row["aspect_ratio"]))
    coordinates = np.array([[row["alpha"], row["theta"], row["aspect_ratio"]]
                            for row in baseline], dtype=float)
    probability = np.linspace(0.0, 1.0, 1025)
    # Two-dimensional (a, u) energy sampler.  Everything the kernel needs from
    # the incoming pair enters through the single scalar
    #     a = lambda1 + lambda3 * z_in + lambda4 * eps,
    # so one table per node over (a, u) represents the memory exactly.  The old
    # one-dimensional table silently dropped lambda3.
    kernel_forms = {row["energy"].get("kernel_form", "conditional_iprojection_v2")
                    for row in baseline}
    if len(kernel_forms) != 1:
        raise ValueError(f"baseline mixes kernel forms: {sorted(kernel_forms)}")
    kernel_form = kernel_forms.pop()
    loss_ceiling = float(max(getattr(bl, "gamma_max", {}).values() or [0.0])) \
        if isinstance(getattr(bl, "gamma_max", {}), dict) else float(getattr(bl, "gamma_max", 0.0))

    eparams = np.array([[row["energy"]["lambda1"], row["energy"]["lambda2"],
                         row["energy"]["lambda3"], row["energy"]["lambda4"]]
                        for row in baseline], dtype=float)
    a_grids, equant = [], []
    for (lambda1, lambda2, lambda3, lambda4), row in zip(eparams, baseline):
        reach = [lambda1, lambda1 + lambda3,
                 lambda1 + lambda4 * loss_ceiling,
                 lambda1 + lambda3 + lambda4 * loss_ceiling]
        low, high = min(reach), max(reach)
        pad = max(0.05 * (high - low), 1.0e-6)
        grid = np.linspace(low - pad, high + pad, ENERGY_A_NODES)
        a_grids.append(grid)
        equant.append(energy_quantile_table(
            lambda3, lambda2, grid, probability, kernel_form=kernel_form))
    a_grids = np.array(a_grids)
    equant = np.array(equant)

    aparams = np.array([[row["angular"]["eta1"], row["angular"]["eta2"]]
                        for row in baseline])
    aquant = np.array([angular_quantiles(parameter, probability) for parameter in aparams])
    energy_errors, angular_errors = [], []
    # Validate the table against the kernel it is meant to represent, at the
    # extremes and centre of each node's own a-range.  Comparing it with the
    # invariant law would be wrong: with memory the conditional mean depends on
    # a, and only the a-averaged law is the invariant one.
    quad = np.linspace(0.0, 1.0, 4097)
    for index, (lambda1, lambda2, lambda3, lambda4) in enumerate(eparams):
        grid = a_grids[index]
        for a in (grid[0], grid[len(grid) // 2], grid[-1]):
            with np.errstate(divide="ignore"):
                logbase = np.log(6.0 * quad * (1.0 - quad))
            if kernel_form == "sinkhorn_bridge_v2":
                logbase = logbase + _bridge_spline(float(lambda3), 256)(quad)
            weight = np.exp(np.clip(logbase + a * quad + lambda2 * quad * quad
                                    - np.max(logbase + a * quad + lambda2 * quad * quad),
                                    -700.0, 700.0))
            weight[0] = weight[-1] = 0.0
            mass = np.trapz(weight, quad)
            exact = (np.trapz(weight * quad, quad) / mass,
                     np.trapz(weight * quad * quad, quad) / mass)
            table = np.array([np.interp(a, grid, equant[index, :, j])
                              for j in range(equant.shape[2])])
            energy_errors.extend((abs(np.trapz(table, probability) - exact[0]),
                                  abs(np.trapz(table * table, probability) - exact[1])))
    for row, quantile in zip(baseline, aquant):
        angular_errors.extend((
            abs(np.trapz(quantile, probability) - row["angular"]["mean_cosine"]),
            abs(np.trapz(0.5 * (3.0 * quantile * quantile - 1.0), probability)
                - row["angular"]["mean_p2"]),
        ))
    sampler_error = max(energy_errors + angular_errors)
    if sampler_error >= 1.0e-3:
        raise ValueError(f"quantile sampler moment error {sampler_error:.3e} exceeds 1e-3")

    coefficient_rows = []
    by_physical_node = defaultdict(list)
    for node in nodes:
        by_physical_node[(node["alpha"], node["theta"], node["aspect_ratio"])].append(node)
    for key, group in sorted(by_physical_node.items()):
        if len(group) <= 1:
            continue
        fitted = fit_lambda1_coefficients(group)
        if not fitted["identifiable"]:
            raise ValueError(f"excitation design is rank deficient at {key}")
        fitted["coordinates"] = list(key)
        coefficient_rows.append(fitted)
    expected_coefficient_nodes = {
        (node["alpha"], node["theta"], node["aspect_ratio"])
        for node in nodes if int(node["ensemble_id"]) != 0
    }
    if expected_coefficient_nodes and len(coefficient_rows) != len(expected_coefficient_nodes):
        raise ValueError("not every excitation node produced an identifiable coefficient fit")
    beta_coordinates = np.array([row["coordinates"] for row in coefficient_rows], dtype=float) \
        if coefficient_rows else np.empty((0, 3))
    beta = np.array([row["beta"] for row in coefficient_rows], dtype=float) \
        if coefficient_rows else np.empty((0, len(FEATURE_NAMES)))
    beta_se = np.array([row["beta_se"] for row in coefficient_rows], dtype=float) \
        if coefficient_rows else np.empty_like(beta)
    beta_deployed = np.array([row["beta_deployed"] for row in coefficient_rows], dtype=bool) \
        if coefficient_rows else np.empty_like(beta, dtype=bool)
    feature_values = np.array([[node["proposal_features"][name] for name in FEATURE_NAMES]
                               for node in nodes])
    diagnostic_values = np.array([[node["proposal_diagnostics"][name]
                                    for name in DIAGNOSTIC_NAMES] for node in nodes])

    uncertainty_names = ("p_exch", "lambda1", "lambda2", "eta1", "eta2")
    uncertainties = np.array([[node.get("uncertainty", {}).get(name, {}).get(
        "standard_error", np.nan) for name in uncertainty_names] for node in baseline])
    joint_deployed = np.array([row["angular"]["joint_deployed"] for row in baseline], dtype=bool)
    joint_parameters = np.full((len(baseline), 3), np.nan)
    for index, row in enumerate(baseline):
        if row["angular"]["joint_parameters"] is not None:
            joint_parameters[index] = row["angular"]["joint_parameters"]

    loss_payload = {"gamma_max": getattr(bl, "gamma_max", {}),
                    "one_hit": getattr(bl, "one_hit", {}),
                    "beta_a": getattr(bl, "beta_a", 1.21),
                    "beta_b": getattr(bl, "beta_b", 3.67)}
    loss_hash = _sha256_bytes(json.dumps(loss_payload, sort_keys=True).encode())
    runtime_root = Path(__file__).resolve().parents[3] / "DSMC_0D_v2/src/dsmc_v2"
    clock_paths = [runtime_root / "ntc.py", runtime_root / "particle.py"]
    clock_payload = b"".join(
        path.name.encode() + b"\0" + path.read_bytes() for path in clock_paths
        if path.is_file())
    clock_hash = _sha256_bytes(clock_payload) if len(clock_payload) else "unavailable"
    stability = _stability_rows(baseline, bl)
    artifact_path = output / "closure_v2.npz"
    np.savez_compressed(
        artifact_path,
        schema_version=np.array(SCHEMA_VERSION), artifact_type=np.array(ARTIFACT_TYPE),
        feature_names=np.array(FEATURE_NAMES), diagnostic_names=np.array(DIAGNOSTIC_NAMES),
        surface_coordinates=coordinates,
        alpha_grid=np.unique(coordinates[:, 0]), theta_grid=np.unique(coordinates[:, 1]),
        aspect_ratio_grid=np.unique(coordinates[:, 2]),
        p_exch=np.array([row["energy"]["p_exch"] for row in baseline]),
        energy_parameters=eparams, angular_parameters=aparams,
        parameter_uncertainties=uncertainties,
        uncertainty_names=np.array(uncertainty_names),
        joint_deployed=joint_deployed, joint_parameters=joint_parameters,
        quantile_probability=probability, energy_quantiles=equant,
        energy_a_grid=a_grids,
        kernel_form=np.array(kernel_form), angular_quantiles=aquant,
        beta_coordinates=beta_coordinates, beta=beta, beta_se=beta_se,
        beta_deployed=beta_deployed,
        feature_lower=np.min(feature_values, axis=0), feature_upper=np.max(feature_values, axis=0),
        diagnostic_lower=np.min(diagnostic_values, axis=0),
        diagnostic_upper=np.max(diagnostic_values, axis=0),
        n_attempts=np.array([row["n_attempts"] for row in baseline], dtype=np.int64),
        n_outcomes=np.array([row["n_outcomes"] for row in baseline], dtype=np.int64),
        ess_fraction=np.array([row["measure"]["ess_fraction"] for row in baseline]),
        loss_hash=np.array(loss_hash), clock_hash=np.array(clock_hash),
        git_sha=np.array(_git_sha()),
        loss_role=np.array("preserved_v1_BL_scalar_loss"),
        clock_role=np.array("preserved_v1_NTC_and_cross_section_polynomial"),
        ctc_target=np.array("surviving_energy_partition_not_absolute_modal_production"),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "file": artifact_path.name,
        "feature_order": list(FEATURE_NAMES),
        "diagnostics": list(DIAGNOSTIC_NAMES),
        "n_runs": len(paths), "n_nodes": len(nodes), "n_baseline_nodes": len(baseline),
        "n_coefficient_nodes": len(coefficient_rows),
        "preserved": ["v1_ntc", "frozen_sigma_c", "BL_scalar_loss", "legacy_runtime_mode"],
        "retired_in_variational_mode": ["conditional_gmm", "rank0_routing", "VSS"],
        "loss_hash": loss_hash, "clock_hash": clock_hash,
        "stability": stability,
        "stability_pass": bool(stability and all(row["unique_stable"] for row in stability)),
        "joint_energy_angle_nodes": int(np.sum(joint_deployed)),
        "maximum_quantile_moment_error": float(sampler_error),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
