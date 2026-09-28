#!/usr/bin/env python3
"""Run one USF trajectory and harvest collision-flux states for exact CTC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

from dsmc_v2.ctc_replay import CollisionFluxReservoir
from dsmc_v2.particle import particle_parameters
from dsmc_v2.simulation import run_simulation


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def row_at(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= index < len(rows) or int(rows[index]["task_id"]) != index:
        raise ValueError("manifest index/task_id mismatch")
    return rows[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--config", default=(
        "DSMC_0D_v2/config/full_domain_baseline_candidate.yaml"))
    args = parser.parse_args()
    row = row_at(Path(args.manifest), args.task)
    if row.get("protocol_version") != "usf-direct-response-v1":
        raise SystemExit("manifest has the wrong response protocol")
    artifact = Path(row["artifact"])
    reference = Path(row["initialization_reference"])
    if (digest(artifact) != row["artifact_sha256"]
            or digest(reference) != row["initialization_reference_sha256"]):
        raise SystemExit("frozen artifact or initialization reference changed")

    prefix = Path(row["output_prefix"])
    success = Path(str(prefix) + "_SUCCESS.json")
    if success.is_file():
        print(f"harvest already complete: {success}")
        return
    prefix.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    ar, alpha = float(row["aspect_ratio"]), float(row["alpha"])
    particles, phi = int(row["particles"]), float(row["volume_fraction"])
    config["particle"]["AR"] = ar
    config["system"].update({
        "alpha": alpha, "kTt": float(row["initial_ttr"]),
        "kTr": float(row["initial_trot"]), "phi": phi,
        "particle_count": particles,
    })
    params = particle_parameters(config)
    box = (particles * params.volume / phi) ** (1.0 / 3.0)
    config["system"]["domain"] = [box] * 3
    if not math.isclose(particles * params.volume / box**3, phi,
                        rel_tol=2.0e-14, abs_tol=2.0e-14):
        raise RuntimeError("volume-fraction construction failed")
    config["time"].update({
        "dt": float(row["dt"]), "dtau": 0.5, "t_end": 1.0e9,
        "tau_end": float(row["tau_end"]), "equilibration_time": 0.0,
    })
    config["flow"] = {"mode": "usf", "shear_rate": float(row["shear_rate"])}
    config["simulation"].update({
        "sphere_collision": False, "exact_initial_temperatures": True,
        "orientation_integrator": "symmetric_midpoint_v1",
        "use_isotropic_eps": True,
    })
    config["microscopic_closure"].update({
        "routing": "variational_v2", "angular": "variational_v2",
        "artifact": str(artifact),
        # Harvest the base trajectory.  Exact CTC labels below determine
        # whether and how a correction is released; DEM never does.
        "invariant_corrections": False,
        "state_update_cpp": 0.05,
        "correction_fallback_gate": "adaptive_base_law",
    })
    windows = [tuple(map(float, token.split(":")))
               for token in row["windows"].split(";")]
    observer = CollisionFluxReservoir(
        Path(str(prefix) + "_replay"), windows,
        int(row["reservoir_capacity"]), int(row["seed"]),
        alpha=alpha, aspect_ratio=ar, mass=params.mass,
        inertia=params.inertia)
    diagnostics = run_simulation(
        config, int(row["seed"]), Path(str(prefix) + ".txt"),
        Path(str(prefix) + "_pressure.txt"),
        Path(str(prefix) + "_orientation.txt"), pair_observer=observer)
    replay = observer.finalize()
    payload = {
        "protocol_version": row["protocol_version"],
        "tag": row["tag"], "manifest": str(Path(args.manifest)),
        "task_id": args.task, "alpha": alpha, "aspect_ratio": ar,
        "initial_branch": row["initial_branch"],
        "artifact": str(artifact), "artifact_sha256": row["artifact_sha256"],
        "initialization_reference_role": "transient_initialization_only_not_fit_target",
        "initialization_reference_sha256": row["initialization_reference_sha256"],
        "particle_count": particles, "reservoirs": replay,
        "simulation_diagnostics": diagnostics,
    }
    temporary = success.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(success)
    print(json.dumps({"task_id": args.task, "n_windows": len(replay),
                      "success": str(success)}, sort_keys=True))


if __name__ == "__main__":
    main()
