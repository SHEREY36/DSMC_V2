"""Build the versioned BL-compatible variational closure artifact."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# numpy 2 renamed trapz; keep one spelling for both
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from dsmc_v2_contracts import DIAGNOSTIC_NAMES, FEATURE_NAMES

from .estimate import NODE_ESTIMATE_CONTRACT, estimate_node
from .fit_exchange import LOGIT_CUBIC_KERNEL, logit_memory_basis
from .fit_coefficients import (
    CORRECTION_PARAMETER_NAMES,
    fit_correction_coefficients,
)
from .projections import (
    _legendre_nodes,
    _bridge_spline,
    adaptive_energy_quantile_table,
    angular_quantiles,
    bridge_mean_map,
    energy_quantile_table,
    incoming_partition_density,
)


SCHEMA_VERSION = "2.4.0"
# Fixed Xi axis for the collision-measure enhancement. Shared by every node so
# the runtime can interpolate the curve like any other surface.
XI_GRID = np.geomspace(0.05, 40.0, 32)
XI_BINS = 24
ENERGY_A_INTERPOLATION_TOLERANCE = 2.0e-4
ENERGY_A_MAX_NODES = 8193
ARTIFACT_TYPE = "bl_variational_closure"
PRECOMPUTE_SCHEMA = "artifact-node-v2-multivariate"
ENERGY_SENSITIVITY_PARAMETERS = ("lambda2", "lambda3")


def _node_memory_shift(energy: dict, z_in) -> np.ndarray:
    """Incoming-state contribution to the conditional natural parameter."""
    z = np.asarray(z_in, dtype=float)
    if energy.get("kernel_form") == LOGIT_CUBIC_KERNEL:
        coefficients = np.asarray(energy["memory_coefficients"], dtype=float)
        basis = logit_memory_basis(
            z, float(energy["memory_center"]), float(energy["memory_scale"]),
            degree=len(coefficients))
        return basis @ coefficients
    return float(energy["lambda3"]) * z


def _measure_enhancement(run_directories, offsets: int = 128) -> np.ndarray:
    """g(Xi) on the shared axis, normalised so THIS node's collision rate holds.

    Two measured facts drive the shape of this function:
      * the enhancement depends on aspect ratio. An AR = 3 curve used at
        AR = 1.1 misses the selected <z> by 0.0186 -- the whole size of the
        effect -- while a per-aspect-ratio curve is out by 0.0002.
      * "normalised to mean one" holds only on the sample it was fitted on.
        The Xi distribution shifts with theta, so <A g>/<A> runs to 1.15 at
        theta = 0.2 and 0.95 at theta = 2 for AR = 3. Dividing by this node's
        own rate multiplier keeps the NTC clock frozen at every grid node.
    """
    from .estimate import _run_propensity
    from .weights import projected_excluded_area
    from dsmc_v2_contracts.io import AI, _vec, load_run

    parts = []
    for directory in run_directories:
        run = load_run(directory)
        propensity = _run_propensity(run, offsets)
        if propensity is None:
            raise ValueError("the collision-measure enhancement requires the "
                             "kinematic propensity; set propensity_offsets")
        values = np.asarray(run.attempts["values"])
        c1, c2 = _vec(values, AI, "c1"), _vec(values, AI, "c2")
        w1, w2 = _vec(values, AI, "omega1"), _vec(values, AI, "omega2")
        speed = np.linalg.norm(c1 - c2, axis=1)
        reach = float(run.metadata["aspect_ratio"]) * float(
            run.metadata.get("diameter", 1.0))
        xi = ((np.linalg.norm(w1, axis=1) + np.linalg.norm(w2, axis=1)) * reach
              / np.maximum(speed, 1.0e-30))
        parts.append((xi, projected_excluded_area(run),
                      np.asarray(propensity, dtype=float)))
    if not parts:
        raise ValueError("no run directories supplied for the enhancement")
    xi = np.concatenate([q[0] for q in parts])
    area = np.concatenate([q[1] for q in parts])
    propensity = np.concatenate([q[2] for q in parts])

    ratio = propensity / np.maximum(area, 1.0e-30)
    edges = np.quantile(xi, np.linspace(0.0, 1.0, XI_BINS + 1))
    which = np.clip(np.digitize(xi, edges[1:-1]), 0, XI_BINS - 1)
    centres = np.array([np.median(xi[which == k]) for k in range(XI_BINS)])
    medians = np.array([np.median(ratio[which == k]) for k in range(XI_BINS)])
    curve = np.interp(XI_GRID, centres, medians)
    applied = np.interp(xi, XI_GRID, curve)
    return curve / (float(np.mean(area * applied)) / float(np.mean(area)))


def _runtime_routing_loss_ceiling(row: dict, bl) -> float:
    """Largest loss covariate that the runtime can hand to this fitted node.

    The BL draw is bounded by ``gamma_max * one_hit_probability``.  Before the
    loss enters lambda4, the runtime rescales it by fitted_mean/runtime_mean.
    Using the largest gamma_max in the *entire* BL table, as the old builder
    did, creates unreachable a-values at nearly elastic nodes and was the main
    reason their tables became both enormous and under-resolved.
    """
    energy = row["energy"]
    if bl is None or float(row["alpha"]) >= 1.0 \
            or not energy.get("loss_covariate_deployed", False):
        return 0.0
    loss = bl.parameters(float(row["alpha"]), float(row["aspect_ratio"]))
    raw_ceiling = float(loss["gamma_max"] * loss["one_hit_probability"])
    runtime_mean = float(loss["mean_loss_fraction"])
    fitted_mean = float(energy["mean_fractional_loss"])
    if runtime_mean > 0.0 and fitted_mean > 0.0:
        raw_ceiling *= fitted_mean / runtime_mean
    return raw_ceiling


def _energy_table_for_node(row: dict, bl, probability: np.ndarray,
                           correction_bounds=(0.0, 0.0)
                           ) -> tuple[np.ndarray, np.ndarray, float]:
    """Compile one node's exact runtime-reachable adaptive energy table."""
    energy = row["energy"]
    memory_grid = np.linspace(1.0e-9, 1.0 - 1.0e-9, 2049)
    memory = _node_memory_shift(energy, memory_grid)
    lambda1 = float(energy["lambda1"])
    lambda4 = float(energy["lambda4"])
    loss_ceiling = _runtime_routing_loss_ceiling(row, bl)
    reach = np.array([
        lambda1 + memory_value + lambda4 * loss_value + correction
        for memory_value in (float(np.min(memory)), float(np.max(memory)))
        for loss_value in (0.0, loss_ceiling)
        for correction in tuple(float(value) for value in correction_bounds)
    ])
    lower, upper = float(np.min(reach)), float(np.max(reach))
    pad = max(0.05 * (upper - lower), 1.0e-6)
    return adaptive_energy_quantile_table(
        float(energy["lambda3"]), float(energy["lambda2"]),
        lower - pad, upper + pad, probability,
        kernel_form=energy.get("kernel_form", "conditional_iprojection_v2"),
        anchor=(energy.get("anchor_c1", 0.0), energy.get("anchor_c2", 0.0)),
        tolerance=ENERGY_A_INTERPOLATION_TOLERANCE,
        max_nodes=ENERGY_A_MAX_NODES)


def _estimate_digest(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def _fit_coefficient_rows(nodes: list[dict]) -> list[dict]:
    rows = []
    grouped = defaultdict(list)
    for node in nodes:
        grouped[(node["alpha"], node["theta"], node["aspect_ratio"])].append(node)
    for key, group in sorted(grouped.items()):
        if len(group) <= 1:
            continue
        fitted = fit_correction_coefficients(group)
        if not fitted["identifiable"]:
            raise ValueError(f"excitation design is rank deficient at {key}")
        if not fitted["linearity_pass"]:
            raise ValueError(
                f"multivariate natural-parameter response is nonlinear at {key}")
        fitted["coordinates"] = list(key)
        rows.append(fitted)
    expected = {(node["alpha"], node["theta"], node["aspect_ratio"])
                for node in nodes if int(node["ensemble_id"]) != 0}
    if expected and len(rows) != len(expected):
        raise ValueError("not every excitation node produced an identifiable coefficient fit")
    return rows


def _correction_spec(nodes: list[dict], coefficient_rows: list[dict]):
    """Return a conservative artifact-wide correction interval and digest."""
    if not coefficient_rows:
        return (0.0, 0.0), "baseline-no-corrections"
    features = np.array([
        [(node.get("cell_features") or node["proposal_features"])[name]
         for name in FEATURE_NAMES] for node in nodes
    ], dtype=float)
    lower, upper = np.min(features, axis=0), np.max(features, axis=0)
    bounds = [0.0]
    for row in coefficient_rows:
        beta = np.asarray(row["beta"], dtype=float) * np.asarray(
            row["beta_deployed"], dtype=bool)
        center = np.asarray(row["feature_center"], dtype=float)
        component_bounds = []
        for component in beta:
            lo = np.where(component >= 0.0, lower - center, upper - center)
            hi = np.where(component >= 0.0, upper - center, lower - center)
            component_bounds.append((float(component @ lo), float(component @ hi)))
        # The table axis contains lambda1 + lambda3*z_in + lambda4*loss.  A
        # conservative unit interval for both covariates bounds every runtime
        # shift without assuming their correlation.
        for z_in in (0.0, 1.0):
            for loss in (0.0, 1.0):
                bounds.extend((component_bounds[0][0]
                               + z_in * component_bounds[2][0]
                               + loss * component_bounds[3][0],
                               component_bounds[0][1]
                               + z_in * component_bounds[2][1]
                               + loss * component_bounds[3][1]))
    payload = json.dumps(coefficient_rows, sort_keys=True).encode()
    return (float(min(bounds)), float(max(bounds))), _sha256_bytes(payload)


def _energy_logit_sensitivities(row: dict, a_grid: np.ndarray,
                                probability: np.ndarray,
                                workers: int = 1) -> np.ndarray:
    """Central differences of logit quantiles at fixed ``a``.

    Changes in lambda1/lambda4 and the ``lambda3*z_in`` part are represented
    exactly by the existing a-axis. These two surfaces represent the remaining
    changes of distribution shape: lambda2 curvature and the Sinkhorn bridge
    potential's lambda3 dependence.
    """
    energy = row["energy"]
    form = energy.get("kernel_form", "conditional_iprojection_v2")
    anchor = (energy.get("anchor_c1", 0.0), energy.get("anchor_c2", 0.0))
    def derivative_for(name: str) -> np.ndarray:
        if name == "lambda3" and form != "sinkhorn_bridge_v2":
            return np.zeros((len(a_grid), len(probability)))
        base = float(energy[name])
        step = max(1.0e-4, 1.0e-3 * (1.0 + abs(base)))
        low_memory = base - step if name == "lambda3" else float(energy["lambda3"])
        high_memory = base + step if name == "lambda3" else float(energy["lambda3"])
        low_lambda2 = base - step if name == "lambda2" else float(energy["lambda2"])
        high_lambda2 = base + step if name == "lambda2" else float(energy["lambda2"])
        low = energy_quantile_table(
            low_memory, low_lambda2, a_grid, probability,
            kernel_form=form, anchor=anchor)
        high = energy_quantile_table(
            high_memory, high_lambda2, a_grid, probability,
            kernel_form=form, anchor=anchor)
        eps = 1.0e-8
        low_logit = np.log(np.clip(low, eps, 1.0 - eps)
                           / (1.0 - np.clip(low, eps, 1.0 - eps)))
        high_logit = np.log(np.clip(high, eps, 1.0 - eps)
                            / (1.0 - np.clip(high, eps, 1.0 - eps)))
        derivative = (high_logit - low_logit) / (2.0 * step)
        derivative[:, (0, -1)] = 0.0
        return derivative

    count = max(1, min(int(workers), len(ENERGY_SENSITIVITY_PARAMETERS)))
    if count == 1:
        values = [derivative_for(name) for name in ENERGY_SENSITIVITY_PARAMETERS]
    else:
        with ThreadPoolExecutor(max_workers=count) as executor:
            values = list(executor.map(derivative_for,
                                       ENERGY_SENSITIVITY_PARAMETERS))
    return np.asarray(values)


def _atomic_savez(path: Path, **arrays) -> None:
    """Publish a complete cache entry atomically on the shared filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def precompute_artifact_node(run_directories, node_estimates, output_directory,
                             index: int, bl, propensity_offsets: int = 128,
                             propensity_workers: int = 1) -> Path:
    """Build the independent geometry and sampler payload for one node.

    This is the unit executed by the Negishi Slurm array.  The final aggregator
    verifies the node coordinate and estimate digest before consuming it, so a
    stale partial from an earlier fit cannot enter the deployed artifact.
    """
    paths = [Path(path) for path in run_directories]
    grouped = defaultdict(list)
    for path in paths:
        grouped[_path_key(path)].append(path)
    nodes = _load_node_estimates(node_estimates, grouped)
    coefficient_rows = _fit_coefficient_rows(nodes)
    correction_bounds, correction_digest = _correction_spec(nodes, coefficient_rows)
    baseline = sorted(
        (node for node in nodes if int(node["ensemble_id"]) == 0),
        key=lambda row: (row["alpha"], row["theta"], row["aspect_ratio"]))
    if not 0 <= int(index) < len(baseline):
        raise IndexError(f"artifact node index {index} outside 0..{len(baseline) - 1}")
    row = baseline[int(index)]
    shards = grouped.get(_node_key(row))
    if not shards:
        raise ValueError(f"no shards for artifact node {_node_key(row)}")

    # Populate the expensive cache using all CPUs assigned to this array task.
    from .estimate import _run_propensity
    from dsmc_v2_contracts.io import load_run
    for shard in shards:
        run = load_run(shard)
        _run_propensity(run, propensity_offsets, workers=propensity_workers)
    enhancement = _measure_enhancement(shards, propensity_offsets)
    probability = np.linspace(0.0, 1.0, 513)
    a_grid, quantiles, interpolation_error = _energy_table_for_node(
        row, bl, probability, correction_bounds)
    sensitivities = _energy_logit_sensitivities(
        row, a_grid, probability, workers=propensity_workers)
    target = Path(output_directory) / f"node_{int(index):04d}.npz"
    _atomic_savez(
        target,
        precompute_schema=np.array(PRECOMPUTE_SCHEMA),
        coordinate=np.array([row["alpha"], row["theta"], row["aspect_ratio"]],
                            dtype=float),
        estimate_digest=np.array(_estimate_digest(row)),
        correction_digest=np.array(correction_digest),
        propensity_offsets=np.array(int(propensity_offsets)),
        quantile_probability=probability,
        energy_a_grid=a_grid,
        energy_quantiles=quantiles,
        energy_logit_sensitivities=sensitivities,
        energy_interpolation_error=np.array(interpolation_error),
        xi_grid=XI_GRID,
        xi_enhancement=enhancement,
    )
    return target


def _load_precomputed_node(directory: Path, index: int, row: dict,
                           probability: np.ndarray, propensity_offsets: int,
                           correction_digest: str = "baseline-no-corrections"):
    path = directory / f"node_{index:04d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact precompute payload {path}")
    with np.load(path, allow_pickle=False) as data:
        if str(data["precompute_schema"]) != PRECOMPUTE_SCHEMA:
            raise ValueError(f"stale precompute schema in {path}")
        expected = np.array([row["alpha"], row["theta"], row["aspect_ratio"]])
        if not np.allclose(data["coordinate"], expected, atol=1.0e-12, rtol=0.0):
            raise ValueError(f"precompute coordinate mismatch in {path}")
        if str(data["estimate_digest"]) != _estimate_digest(row):
            raise ValueError(f"precompute estimate digest mismatch in {path}")
        cached_correction = (str(data["correction_digest"])
                             if "correction_digest" in data.files
                             else "baseline-no-corrections")
        if cached_correction != correction_digest:
            raise ValueError(f"precompute correction-fit digest mismatch in {path}")
        if int(data["propensity_offsets"]) != int(propensity_offsets):
            raise ValueError(f"precompute propensity resolution mismatch in {path}")
        if not np.array_equal(data["quantile_probability"], probability) \
                or not np.array_equal(data["xi_grid"], XI_GRID):
            raise ValueError(f"precompute grid mismatch in {path}")
        grid = np.asarray(data["energy_a_grid"], dtype=float)
        quantiles = np.asarray(data["energy_quantiles"], dtype=float)
        sensitivities = np.asarray(data["energy_logit_sensitivities"], dtype=float)
        enhancement = np.asarray(data["xi_enhancement"], dtype=float)
        error = float(data["energy_interpolation_error"])
    if grid.ndim != 1 or quantiles.shape != (len(grid), len(probability)) \
            or sensitivities.shape != (len(ENERGY_SENSITIVITY_PARAMETERS),
                                       len(grid), len(probability)) \
            or enhancement.shape != XI_GRID.shape \
            or not np.all(np.diff(grid) > 0.0) \
            or not np.all(np.diff(quantiles, axis=1) >= 0.0):
        raise ValueError(f"invalid precompute array shapes/order in {path}")
    if not np.isfinite(error) or error > ENERGY_A_INTERPOLATION_TOLERANCE:
        raise ValueError(f"energy interpolation error {error:.3e} in {path}")
    return grid, quantiles, sensitivities, enhancement, error


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
    directories = ([directory] if isinstance(directory, (str, os.PathLike))
                   else list(directory))
    paths = [path for item in directories
             for path in sorted(Path(item).glob("alpha_*.json"))]
    nodes = [json.loads(path.read_text()) for path in paths]
    keys = {_node_key(node) for node in nodes}
    if len(nodes) != len(keys):
        raise ValueError("precomputed node estimates contain duplicate grid/ensemble keys")
    baseline_keys = {_node_key(node) for node in nodes
                     if int(node.get("ensemble_id", 0)) == 0}
    if baseline_keys != set(expected_groups):
        missing = sorted(set(expected_groups) - baseline_keys)
        extra = sorted(baseline_keys - set(expected_groups))
        raise ValueError(f"precomputed baseline grid mismatch; missing={missing}, extra={extra}")
    for node in nodes:
        if node.get("estimator_contract") != NODE_ESTIMATE_CONTRACT:
            raise ValueError(
                f"stale node estimate for {_node_key(node)}: estimator contract "
                f"{node.get('estimator_contract')!r}, expected {NODE_ESTIMATE_CONTRACT!r}")
        missing_fields = [name for name in (
            "cell_features", "incoming_law", "incoming_law_energy") if name not in node]
        if missing_fields:
            raise ValueError(
                f"stale node estimate for {_node_key(node)}: missing "
                + ", ".join(missing_fields))
        # Compare shard identities, not absolute paths: a grid estimated on the
        # cluster must validate against the same shards copied to another root.
        # The directory name carries alpha, theta, AR, ensemble and shard, so
        # this still catches an estimate built from different shards.
        key = _node_key(node)
        lookup_key = key
        if int(node.get("ensemble_id", 0)) != 0 and node.get("excitation"):
            # Importance-sampled excitations are virtual ensembles evaluated on
            # the baseline shard.  They intentionally have no fresh CTC
            # directory bearing their new ensemble ID.
            lookup_key = (key[0], key[1], key[2], 0)
        if lookup_key not in expected_groups:
            raise ValueError(f"node {key} has no matching baseline or direct CTC shard")
        expected = {Path(path).name for path in expected_groups[lookup_key]}
        actual = {Path(path).name for path in node.get("source_runs", [])}
        if actual != expected:
            raise ValueError(
                f"stale node estimate for {_node_key(node)}: "
                f"expected shards {sorted(expected)}, got {sorted(actual)}")
        qa = node.get("qa", {})
        if int(node.get("ensemble_id", 0)) == 0:
            passed = qa.get("precision_pass", qa.get("sentinel_pass", False))
        else:
            # A virtual excitation is a response-design point. Its individual
            # lambda1 half-width need not meet the production baseline target;
            # overlap, all physical sentinels, and the held-out response fit
            # are the relevant gates.
            amplitude = abs(float(node.get("excitation", {}).get("eta", 0.0)))
            heldout_model_form_only = (
                amplitude > 0.25 + 1.0e-12
                and set(qa.get("continuation_reasons", [])) == {"model_form"}
                and all(qa.get(name, False) for name in (
                    "angular_projection_pass", "elastic_pass",
                    "energy_projection_pass", "ess_pass",
                    "incoming_partition_pass", "memory_diagnostic_pass",
                    "propensity_pass", "proposal_balance_pass")))
            passed = (node.get("excitation_status", "pass") == "pass"
                      and node.get("excitation", {}).get("usable", True)
                      and (qa.get("sentinel_pass", qa.get("precision_pass", False))
                           or heldout_model_form_only))
        if not passed:
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


def _stability_rows(baseline: list[dict], bl=None, sampler=None) -> list[dict]:
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
        # The gate must evaluate the actual node-owned conditional sampler and
        # average it over each node's OWN incoming law.
        anchor1 = PchipInterpolator(theta, [row["energy"].get("anchor_c1", 0.0)
                                            for row in nodes])
        anchor2 = PchipInterpolator(theta, [row["energy"].get("anchor_c2", 0.0)
                                            for row in nodes])
        if any("incoming_law_energy" not in row for row in nodes):
            # Defaulting to Beta(2,2) here would make the incoming law
            # theta-independent, which silently removes the very theta
            # dependence the drift balance is measuring.
            raise ValueError(
                "the stability gate needs each node's ENERGY-weighted incoming "
                "law; re-estimate the grid so incoming_law_energy is present")
        incoming1 = PchipInterpolator(
            theta, [row["incoming_law_energy"]["c1"] for row in nodes])
        incoming2 = PchipInterpolator(
            theta, [row["incoming_law_energy"]["c2"] for row in nodes])
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
        # The hybrid DSMC has deliberately separate loss variables.  The BL
        # draw controls the amount of energy destroyed, while the bridge must
        # see a covariate on the scale of the CTC loss against which lambda4
        # was fitted.  Feeding the BL mean to both jobs moves the inelastic
        # root whenever those means differ (13--28 percent on the sentinel).
        fitted_loss = PchipInterpolator(
            theta, [row["energy"].get("mean_fractional_loss", mean_loss)
                    for row in nodes])
        sampler_means = {}
        if sampler is not None:
            probability = sampler["probability"]
            for node in nodes:
                node_key = (float(node["alpha"]), float(node["theta"]),
                            float(node["aspect_ratio"]))
                a_axis, quantiles = sampler["nodes"][node_key]
                sampler_means[node_key] = (
                    a_axis, _trapezoid(quantiles, probability, axis=1))

        def _incoming_mass(value):
            """The node's measured incoming law, as fitted under the energy weight."""
            log = (np.log(quad * 6.0 * grid * (1.0 - grid))
                   + float(incoming1(value)) * grid
                   + float(incoming2(value)) * grid * grid)
            mass = np.exp(log - log.max())
            return mass / np.sum(mass)

        def post_collision_partition(value):
            """<E z'>/<E> of the exact law exported to the DSMC runtime."""
            mass = _incoming_mass(value)
            if sampler is None:
                # Analytic fallback retained for unit tests and diagnostics
                # that operate on node estimates before tables are generated.
                parameters = np.array([float(lambda3(value)), float(lambda1(value)),
                                       float(lambda2(value)), float(lambda4(value))])
                anchor = (float(anchor1(value)), float(anchor2(value)))
                mean_map = bridge_mean_map(
                    parameters, grid, float(fitted_loss(value)), 192, anchor)
                return float(mass @ mean_map), float(mass @ grid)

            # The runtime blends neighbouring fitted conditional quantiles. It must
            # not first mix their nonlinear natural parameters, a-grids and
            # quantile tables: those operations do not commute and the old
            # order created three roots where the CTC operator has one.
            exact_theta = np.flatnonzero(np.isclose(theta, value, atol=1.0e-12,
                                                    rtol=0.0))
            if len(exact_theta):
                stencil = [(int(exact_theta[0]), 1.0)]
            else:
                upper = int(np.searchsorted(theta, value))
                if upper == 0 or upper == len(theta):
                    raise ValueError("stability query lies outside its theta grid")
                lower = upper - 1
                high_weight = float((value - theta[lower])
                                    / (theta[upper] - theta[lower]))
                stencil = [(lower, 1.0 - high_weight), (upper, high_weight)]
            mean_map = np.zeros_like(grid)
            for index, physical_weight in stencil:
                node = nodes[index]
                node_key = (float(node["alpha"]), float(node["theta"]),
                            float(node["aspect_ratio"]))
                a_axis, conditional_mean = sampler_means[node_key]
                energy = node["energy"]
                route_loss = float(energy.get("mean_fractional_loss", mean_loss))
                a = (float(energy["lambda1"])
                     + _node_memory_shift(energy, grid)
                     + float(energy.get("lambda4", 0.0)) * route_loss)
                mean_map += physical_weight * np.interp(a, a_axis, conditional_mean)
            return float(mass @ mean_map), float(mass @ grid)

        def drift(value):
            post_partition, incoming = post_collision_partition(value)
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
                     "routing_loss_at_root": (None if len(roots) != 1 else
                                              float(fitted_loss(roots[0]))),
                     "drift_model": (
                         "node_first_quantile_interpolation_with_CTC_routing_loss_"
                         "plus_frozen_BL_budget_loss" if sampler is not None else
                         "analytic_bridge_with_CTC_routing_loss_"
                         "plus_frozen_BL_budget_loss"),
                     "includes_surface_derivatives": True})
    return rows


def build_artifact(run_directories, output_directory, bl=None,
                   n_bootstrap: int = 200, node_estimates=None,
                   propensity_offsets: int = 128,
                   precomputed_directory=None) -> dict:
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
    coefficient_rows = _fit_coefficient_rows(nodes)
    correction_bounds, correction_digest = _correction_spec(nodes, coefficient_rows)
    coordinates = np.array([[row["alpha"], row["theta"], row["aspect_ratio"]]
                            for row in baseline], dtype=float)
    # 513 nodes reproduce the kernel's first two moments to ~1e-5, and the
    # a-axis is the dimension that has to be wide.
    probability = np.linspace(0.0, 1.0, 513)
    # Two-dimensional (a, u) energy sampler. Everything the kernel needs from
    # the incoming pair enters through one scalar
    #     a = lambda1 + memory(z_in) + lambda4 * eps.
    # ``memory`` is linear for the bridge and bounded-logit cubic at repaired
    # nodes. One table per node over (a, u) therefore represents either form
    # exactly; the old one-dimensional table silently dropped this dependence.
    kernel_forms = np.array([
        row["energy"].get("kernel_form", "conditional_iprojection_v2")
        for row in baseline
    ])
    kernel_form_label = (str(kernel_forms[0]) if len(set(kernel_forms.tolist())) == 1
                         else "nodewise_mixed_v1")
    eparams = np.array([[row["energy"]["lambda1"], row["energy"]["lambda2"],
                         row["energy"]["lambda3"], row["energy"]["lambda4"]]
                        for row in baseline], dtype=float)
    a_grids, equant, energy_sensitivities = [], [], []
    enhancement, interpolation_errors = [], []
    precomputed = None if precomputed_directory is None else Path(precomputed_directory)
    for index, row in enumerate(baseline):
        if precomputed is not None:
            grid, quantile, sensitivity, curve, interpolation_error = _load_precomputed_node(
                precomputed, index, row, probability, propensity_offsets,
                correction_digest)
        else:
            grid, quantile, interpolation_error = _energy_table_for_node(
                row, bl, probability, correction_bounds)
            sensitivity = _energy_logit_sensitivities(row, grid, probability)
            shards = dict(grouped).get(_node_key(row))
            if not shards:
                raise ValueError(f"no shards for node {_node_key(row)}; cannot "
                                 "measure its collision-measure enhancement")
            curve = _measure_enhancement(shards, propensity_offsets)
        a_grids.append(grid)
        equant.append(quantile)
        energy_sensitivities.append(sensitivity)
        enhancement.append(curve)
        interpolation_errors.append(interpolation_error)
    enhancement = np.asarray(enhancement)

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
        kernel_form = kernel_forms[index]
        for a in (grid[0], grid[len(grid) // 2], grid[-1]):
            with np.errstate(divide="ignore"):
                logbase = np.log(6.0 * quad * (1.0 - quad))
            anchor = (baseline[index]["energy"].get("anchor_c1", 0.0),
                      baseline[index]["energy"].get("anchor_c2", 0.0))
            if kernel_form == "sinkhorn_bridge_v2":
                logbase = (logbase + anchor[0] * quad + anchor[1] * quad * quad
                           + _bridge_spline(float(lambda3), 256, anchor)(quad))
            weight = np.exp(np.clip(logbase + a * quad + lambda2 * quad * quad
                                    - np.max(logbase + a * quad + lambda2 * quad * quad),
                                    -700.0, 700.0))
            weight[0] = weight[-1] = 0.0
            mass = _trapezoid(weight, quad)
            exact = (_trapezoid(weight * quad, quad) / mass,
                     _trapezoid(weight * quad * quad, quad) / mass)
            table = np.array([np.interp(a, grid, equant[index][:, j])
                              for j in range(len(probability))])
            energy_errors.extend((abs(_trapezoid(table, probability) - exact[0]),
                                  abs(_trapezoid(table * table, probability) - exact[1])))
    for row, quantile in zip(baseline, aquant):
        angular_errors.extend((
            abs(_trapezoid(quantile, probability) - row["angular"]["mean_cosine"]),
            abs(_trapezoid(0.5 * (3.0 * quantile * quantile - 1.0), probability)
                - row["angular"]["mean_p2"]),
        ))
    energy_sampler_error = max(energy_errors)
    angular_sampler_error = max(angular_errors)
    sampler_error = max(energy_sampler_error, angular_sampler_error)
    if sampler_error >= 1.0e-3:
        raise ValueError(f"quantile sampler moment error {sampler_error:.3e} exceeds 1e-3")

    beta_coordinates = np.array([row["coordinates"] for row in coefficient_rows], dtype=float) \
        if coefficient_rows else np.empty((0, 3))
    beta = np.array([row["beta"] for row in coefficient_rows], dtype=float) \
        if coefficient_rows else np.empty((0, len(CORRECTION_PARAMETER_NAMES),
                                            len(FEATURE_NAMES)))
    beta_se = np.array([row["beta_se"] for row in coefficient_rows], dtype=float) \
        if coefficient_rows else np.empty_like(beta)
    beta_deployed = np.array([row["beta_deployed"] for row in coefficient_rows], dtype=bool) \
        if coefficient_rows else np.empty_like(beta, dtype=bool)
    beta_feature_center = np.array([row["feature_center"] for row in coefficient_rows],
                                   dtype=float) \
        if coefficient_rows else np.empty((0, len(FEATURE_NAMES)))
    # Runtime features are cell moments.  Collision-attempt moments are flux
    # weighted (a Maxwellian reports a2_tr ~= -0.032 there), so using them as
    # the runtime hull makes every real cell out of domain even at startup.
    feature_values = np.array([
        [(node.get("cell_features") or node["proposal_features"])[name]
         for name in FEATURE_NAMES]
        for node in nodes
    ])
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
    sampler = {
        "probability": probability,
        "nodes": {
            (float(row["alpha"]), float(row["theta"]),
             float(row["aspect_ratio"])): (a_grids[index], equant[index])
            for index, row in enumerate(baseline)
        },
    }
    stability = _stability_rows(baseline, bl, sampler=sampler)
    if not stability or not all(row["unique_stable"] for row in stability):
        # Fail closed. A kernel with no root, or more than one, has no steady
        # temperature ratio to deploy, and exporting it anyway is how a
        # spurious fixed point reaches a production run.
        detail = []
        for row in stability:
            if row["unique_stable"]:
                continue
            if not row["roots"]:
                why = ("no root inside the calibrated theta range: the fixed "
                       "point lies outside the grid, so generate CTC nodes that "
                       "bracket it")
            else:
                why = f"{len(row['roots'])} roots {row['roots']}: not a unique attractor"
            detail.append(f"(alpha={row['alpha']}, AR={row['aspect_ratio']}) {why}")
        raise ValueError("closure has no unique stable temperature ratio -- "
                         + "; ".join(detail or ["no nodes to test"]))
    artifact_path = output / "closure_v2.npz"
    energy_offsets = np.r_[0, np.cumsum([len(grid) for grid in a_grids])].astype(np.int64)
    energy_a_packed = np.concatenate(a_grids)
    energy_quantiles_packed = np.concatenate(equant, axis=0)
    energy_sensitivities_packed = np.concatenate(
        [np.moveaxis(value, 0, 1) for value in energy_sensitivities], axis=0)
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
        xi_grid=XI_GRID, xi_enhancement=enhancement,
        quantile_probability=probability, energy_quantiles=energy_quantiles_packed,
        energy_logit_sensitivities=energy_sensitivities_packed,
        energy_sensitivity_parameter_names=np.array(ENERGY_SENSITIVITY_PARAMETERS),
        energy_a_grid=energy_a_packed, energy_a_offsets=energy_offsets,
        energy_anchor=np.array([[row["energy"].get("anchor_c1", 0.0),
                                 row["energy"].get("anchor_c2", 0.0)]
                                for row in baseline], dtype=float),
        energy_mean_loss=np.array([row["energy"]["mean_fractional_loss"]
                                   for row in baseline], dtype=float),
        energy_interpolation=np.array("node_first_quantile_interpolation_v1"),
        kernel_form=np.array(kernel_form_label), angular_quantiles=aquant,
        energy_kernel_forms=kernel_forms,
        energy_memory_coefficients=np.array([
            row["energy"].get("memory_coefficients",
                              [row["energy"]["lambda3"], 0.0, 0.0])
            for row in baseline], dtype=float),
        energy_memory_center=np.array([
            row["energy"].get("memory_center", 0.0) for row in baseline], dtype=float),
        energy_memory_scale=np.array([
            row["energy"].get("memory_scale", 1.0) for row in baseline], dtype=float),
        correction_parameter_names=np.array(CORRECTION_PARAMETER_NAMES),
        beta_coordinates=beta_coordinates, beta=beta, beta_se=beta_se,
        beta_deployed=beta_deployed, beta_feature_center=beta_feature_center,
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
        "coefficient_fit": "shared_baseline_gls_central_amplitudes_multivariate_v2",
        "correction_parameters": list(CORRECTION_PARAMETER_NAMES),
        "correction_bounds": list(correction_bounds),
        "correction_digest": correction_digest,
        "coefficient_validation_relative_rmse_max": (
            max(row["maximum_validation_relative_rmse"] for row in coefficient_rows)
            if coefficient_rows else None),
        "preserved": ["v1_ntc", "frozen_sigma_c", "BL_scalar_loss", "legacy_runtime_mode"],
        "retired_in_variational_mode": ["conditional_gmm", "rank0_routing", "VSS"],
        "loss_hash": loss_hash, "clock_hash": clock_hash,
        "stability": stability,
        "stability_pass": bool(stability and all(row["unique_stable"] for row in stability)),
        "joint_energy_angle_nodes": int(np.sum(joint_deployed)),
        "kernel_forms": sorted(set(kernel_forms.tolist())),
        "maximum_quantile_moment_error": float(sampler_error),
        "maximum_energy_quantile_moment_error": float(energy_sampler_error),
        "maximum_angular_quantile_moment_error": float(angular_sampler_error),
        "energy_table_layout": "packed_adaptive_v1",
        "energy_table_rows": int(len(energy_a_packed)),
        "energy_shape_response": "logit_quantile_tangent_lambda2_lambda3_v1",
        "maximum_energy_interpolation_error": float(max(interpolation_errors)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
