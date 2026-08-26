"""Build and validate the versioned collision_operator_v2 artifact bundle."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import load_run

from .energy_library import build_energy_library
from .estimate import estimate_node
from .pair_clock import fit_pair_clock
from .surfaces import fit_surface, transformed_coordinates


def _node_key(run) -> tuple[float, float, float]:
    return (float(run.metadata["alpha"]), float(run.metadata["theta"]),
            float(run.metadata["aspect_ratio"]))


def build_artifact(run_directories, output_directory, n_bootstrap: int = 2000) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    runs = [load_run(path) for path in run_directories]
    if not runs:
        raise ValueError("no CTC runs supplied")
    grouped = defaultdict(list)
    for path, run in zip(run_directories, runs):
        grouped[_node_key(run)].append(path)
    nodes = [estimate_node(paths, n_bootstrap=n_bootstrap) for _, paths in sorted(grouped.items())]
    (output / "node_estimates_v2.json").write_text(json.dumps(nodes, indent=2, sort_keys=True) + "\n")

    # Geometry is independent of restitution. Prefer one elastic theta=1 run
    # per AR so common-random alpha duplicates do not receive extra weight.
    clock_models = {}
    for ar in sorted({_node_key(run)[2] for run in runs}):
        subset = [run for run in runs if np.isclose(_node_key(run)[2], ar)]
        reference = [run for run in subset if np.isclose(_node_key(run)[0], 1.0)
                     and np.isclose(_node_key(run)[1], 1.0)]
        chosen = reference or [subset[0]]
        clock_models[f"{ar:.12g}"] = fit_pair_clock(chosen).to_dict()
    (output / "pair_clock_v2.json").write_text(json.dumps(clock_models, indent=2, sort_keys=True) + "\n")

    inelastic = [run for run in runs if float(run.metadata["alpha"]) < 1.0]
    if not inelastic:
        raise ValueError("at least one inelastic run is required for the energy library")
    build_energy_library(inelastic).save(str(output / "energy_library_v2.npz"))

    vss_rows = []
    for node in nodes:
        if np.isclose(node["theta"], 1.0):
            if not node["qa"]["vss_representable"]:
                raise ValueError(
                    f"VSS rank-2 target is not representable at alpha={node['alpha']}, "
                    f"AR={node['aspect_ratio']}"
                )
            vss_rows.append({
                "alpha": node["alpha"], "aspect_ratio": node["aspect_ratio"],
                "B2": node["quantities"]["B2"],
                "alpha_eff": node["quantities"]["alpha_eff"],
            })
    if not vss_rows:
        raise ValueError("VSS estimation requires theta=1 runs")
    theta_diagnostics = []
    theta_independence_pass = True
    references = {(row["alpha"], row["aspect_ratio"]): row for row in nodes
                  if np.isclose(row["theta"], 1.0)}
    for node in nodes:
        reference = references.get((node["alpha"], node["aspect_ratio"]))
        if reference is None or np.isclose(node["theta"], 1.0):
            continue
        actual = node["quantities"]["B2"]
        baseline = reference["quantities"]["B2"]
        combined_se = np.hypot(actual["standard_error"] or 0.0,
                               baseline["standard_error"] or 0.0)
        difference = abs(actual["estimate"] - baseline["estimate"])
        passed = difference <= 0.02 + 3.0 * combined_se
        theta_independence_pass &= passed
        theta_diagnostics.append({
            "alpha": node["alpha"], "aspect_ratio": node["aspect_ratio"],
            "theta": node["theta"], "B2": actual["estimate"],
            "reference_B2": baseline["estimate"], "absolute_difference": difference,
            "combined_standard_error": combined_se, "pass": passed,
        })
    (output / "vss_theta_diagnostics_v2.json").write_text(json.dumps({
        "schema_version": "2.0.0", "absolute_tolerance": 0.02,
        "sigma_multiplier": 3.0, "pass": theta_independence_pass,
        "rows": theta_diagnostics,
    }, indent=2, sort_keys=True) + "\n")
    if not theta_independence_pass:
        raise ValueError("held-out theta runs reject the temperature-independent VSS assumption")
    (output / "vss_rank2_v2.json").write_text(json.dumps({
        "schema_version": "2.0.0", "inputs": ["alpha", "aspect_ratio"],
        "forbidden_inputs": ["theta", "energy", "f_tr", "dissipation"],
        "rows": vss_rows,
    }, indent=2, sort_keys=True) + "\n")

    surface_names = [name for name in nodes[0]["quantity_order"]
                     if name not in ("B2", "alpha_eff")]
    surfaces = {}
    if len(nodes) >= 8 and len({n["alpha"] for n in nodes}) >= 2 \
            and len({n["theta"] for n in nodes}) >= 2 \
            and len({n["aspect_ratio"] for n in nodes}) >= 2:
        coordinates = transformed_coordinates(
            np.array([n["alpha"] for n in nodes]), np.array([n["theta"] for n in nodes]),
            np.array([n["aspect_ratio"] for n in nodes]))
        for name in surface_names:
            value = np.array([n["quantities"][name]["estimate"] for n in nodes])
            error = np.array([n["quantities"][name]["standard_error"] or np.nan for n in nodes])
            try:
                surfaces[name] = fit_surface(coordinates, value,
                    ["one_minus_alpha_squared", "log_theta", "log_AR"], error).to_dict()
            except ValueError:
                continue
    (output / "reduced_surfaces_v2.json").write_text(json.dumps({
        "schema_version": "2.0.0", "surfaces": surfaces,
        "runtime_constraints": {"Gamma_at_alpha_1": 0.0, "no_extrapolation": True},
    }, indent=2, sort_keys=True) + "\n")

    manifest = {
        "schema_version": "2.0.0", "artifact_type": "collision_operator_v2",
        "runtime_modes": ["pair_resolved", "moment16"],
        "angular_model": "vss_rank2", "p_eta": None,
        "n_runs": len(runs), "n_nodes": len(nodes),
        "files": ["pair_clock_v2.json", "energy_library_v2.npz",
                  "vss_rank2_v2.json", "vss_theta_diagnostics_v2.json", "reduced_surfaces_v2.json",
                  "node_estimates_v2.json"],
        "reduced_ready": bool(surfaces),
        "dem_calibration_used": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
