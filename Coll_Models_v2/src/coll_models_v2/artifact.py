"""Build the conservative microscopic_closure_v2 artifact bundle."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES

from .direction_library import build_direction_library_from_paths
from .estimate import estimate_node
from .legacy_bl import LegacyBL
from .surfaces import fit_surface, transformed_coordinates


def _node_key(run) -> tuple[float, float, float]:
    values = run if isinstance(run, dict) else run.metadata
    return (float(values["alpha"]), float(values["theta"]),
            float(values["aspect_ratio"]))


def _fit_routing_surfaces(nodes: list[dict]) -> dict:
    inelastic = [node for node in nodes if node["alpha"] < 1.0]
    names = (["F0", "C_M", "F_C", "total_loss_compatibility_ratio"]
             + [f"beta_{name}" for name in FEATURE_NAMES]
             + [f"beta_ctc_{name}" for name in FEATURE_NAMES])
    surfaces = {}
    if (len(inelastic) >= 8 and len({n["alpha"] for n in inelastic}) >= 2
            and len({n["theta"] for n in inelastic}) >= 2
            and len({n["aspect_ratio"] for n in inelastic}) >= 2):
        coordinates = transformed_coordinates(
            np.array([n["alpha"] for n in inelastic]),
            np.array([n["theta"] for n in inelastic]),
            np.array([n["aspect_ratio"] for n in inelastic]))
        for name in names:
            values = np.array([n["quantities"][name]["estimate"] for n in inelastic])
            errors = np.array([n["quantities"][name]["standard_error"] or np.nan
                               for n in inelastic])
            try:
                surfaces[name] = fit_surface(
                    coordinates, values,
                    ["one_minus_alpha_squared", "log_theta", "log_AR"], errors).to_dict()
            except ValueError:
                pass
    return surfaces


def _path_key(path) -> tuple[float, float, float]:
    metadata = json.loads((Path(path) / "metadata_v2.json").read_text())
    return (float(metadata["alpha"]), float(metadata["theta"]),
            float(metadata["aspect_ratio"]))


def _load_node_estimates(directory, expected_groups) -> list[dict]:
    nodes = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("alpha_*.json"))]
    keys = {_node_key(node) for node in nodes}
    if len(nodes) != len(keys):
        raise ValueError("precomputed node estimates contain duplicate keys")
    if keys != set(expected_groups):
        missing = sorted(set(expected_groups) - keys)
        extra = sorted(keys - set(expected_groups))
        raise ValueError(f"precomputed node grid mismatch; missing={missing}, extra={extra}")
    for node in nodes:
        key = _node_key(node)
        expected = {Path(path).resolve() for path in expected_groups[key]}
        actual = {Path(path).resolve() for path in node.get("source_runs", [])}
        if actual != expected:
            raise ValueError(f"stale node estimate for {key}; expected shards={sorted(expected)}, "
                             f"estimated shards={sorted(actual)}")
    failed = [node for node in nodes if not node.get("qa", {}).get("precision_pass", False)]
    if failed:
        raise ValueError(f"{len(failed)} precomputed nodes have not passed precision QA")
    return sorted(nodes, key=_node_key)


def build_artifact(run_directories, output_directory, bl: LegacyBL,
                   n_bootstrap: int = 2000, node_estimates=None) -> dict:
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
    surfaces = _fit_routing_surfaces(nodes)
    routing_payload = {
        "schema_version": "2.1.0", "artifact_type": "routing16_v2",
        "feature_order": list(FEATURE_NAMES),
        "coordinates": ["one_minus_alpha_squared", "log_theta", "log_AR"],
        "runtime_equation": "F_tr=F_C*(1+sum(beta_ctc_a*X_a))",
        "routing_range": "unbounded_modal_production_ratio_not_a_probability",
        "cross_section_role": "reported_audit_only_frozen_v1_clock_is_authoritative",
        "total_loss_kernel": "preserved_v1_BL_gamma_max_times_P1hit_times_Beta(1.21,3.67)",
        "total_loss_compatibility_role": "reported_audit_not_a_routing_normalization",
        "design_hull": {"alpha": [0.5, 0.99], "theta": [0.1, 1.2],
                        "aspect_ratio": [1.1, 3.0]},
        "nodes": nodes, "surfaces": surfaces,
    }
    (output / "routing16_v2.json").write_text(
        json.dumps(routing_payload, indent=2, sort_keys=True) + "\n")

    vss_rows = []
    for node in nodes:
        if not np.isclose(node["theta"], 1.0):
            continue
        if not node["qa"]["vss_representable"]:
            raise ValueError(f"unrepresentable VSS target at alpha={node['alpha']}, "
                             f"AR={node['aspect_ratio']}")
        vss_rows.append({
            "alpha": node["alpha"], "aspect_ratio": node["aspect_ratio"],
            **{name: node["quantities"][name]
               for name in ("B2", "alpha_eff", "mean_P1", "mean_P2", "mean_P3", "mean_P4")},
        })
    if not vss_rows:
        raise ValueError("VSS export requires theta=1 CTC runs")
    references = {(row["alpha"], row["aspect_ratio"]): row for row in nodes
                  if np.isclose(row["theta"], 1.0)}
    theta_diagnostics = []
    theta_pass = True
    for node in nodes:
        if np.isclose(node["theta"], 1.0):
            continue
        reference = references.get((node["alpha"], node["aspect_ratio"]))
        if reference is None:
            continue
        actual, baseline = node["quantities"]["B2"], reference["quantities"]["B2"]
        combined_se = float(np.hypot(actual["standard_error"] or 0.0,
                                     baseline["standard_error"] or 0.0))
        difference = abs(actual["estimate"] - baseline["estimate"])
        passed = difference <= 0.02 + 3.0 * combined_se
        theta_pass &= passed
        theta_diagnostics.append({
            "alpha": node["alpha"], "theta": node["theta"],
            "aspect_ratio": node["aspect_ratio"], "B2": actual["estimate"],
            "theta1_B2": baseline["estimate"], "absolute_difference": difference,
            "combined_standard_error": combined_se, "pass": bool(passed),
        })
    if not theta_pass:
        raise ValueError("held-out theta runs reject temperature-independent VSS")
    vss_points = np.array([[1.0 - row["alpha"]**2, np.log(row["aspect_ratio"])]
                           for row in vss_rows])
    vss_surfaces = {}
    if len(vss_rows) >= 4 and len(np.unique(vss_points[:, 0])) >= 2 \
            and len(np.unique(vss_points[:, 1])) >= 2:
        for name in ("B2", "alpha_eff"):
            values = np.array([row[name]["estimate"] for row in vss_rows])
            errors = np.array([row[name]["standard_error"] or np.nan for row in vss_rows])
            vss_surfaces[name] = fit_surface(
                vss_points, values, ["one_minus_alpha_squared", "log_AR"], errors).to_dict()
    (output / "vss_rank2_v2.json").write_text(json.dumps({
        "schema_version": "2.1.0", "artifact_type": "vss_rank2_v2",
        "inputs": ["alpha", "aspect_ratio"],
        "forbidden_inputs": ["theta", "energy", "F_tr", "dissipation", "p_eta"],
        "fit_target": "mean(1-P2(ghat_pre dot ghat_post))",
        "rows": vss_rows, "surfaces": vss_surfaces,
        "theta_independence_diagnostics": {"absolute_tolerance": 0.02,
            "sigma_multiplier": 3.0, "pass": bool(theta_pass), "rows": theta_diagnostics},
    }, indent=2, sort_keys=True) + "\n")

    inelastic_paths = [path for key, shards in grouped.items() if key[0] < 1.0
                       for path in shards]
    direction = build_direction_library_from_paths(inelastic_paths)
    direction.save(str(output / "rotational_direction_v2.npz"))
    manifest = {
        "schema_version": "2.1.0", "artifact_type": "microscopic_closure_v2",
        "production_changes": ["dissipation_routing", "angular_scattering"],
        "preserved": ["v1_ntc", "frozen_sigma_c", "conditional_gmm", "BL_total_loss",
                      "Zr", "reservoir_clipping", "time_integration", "outputs"],
        "files": ["routing16_v2.json", "vss_rank2_v2.json",
                  "rotational_direction_v2.npz"],
        "n_runs": len(paths), "n_nodes": len(nodes),
        "dem_calibration_used": False, "p_eta": None,
        "pair_clock_exported": False, "energy_kernel_exported": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
