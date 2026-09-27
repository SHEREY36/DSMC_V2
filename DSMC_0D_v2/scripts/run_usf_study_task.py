#!/usr/bin/env python3
"""Run one high-statistics, frozen-artifact USF study realization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

from dsmc_v2.particle import particle_parameters
from dsmc_v2.simulation import run_simulation


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def manifest_row(path: Path, task: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= task < len(rows):
        raise IndexError(f"task {task} outside manifest with {len(rows)} rows")
    row = rows[task]
    if int(row["task_id"]) != task:
        raise ValueError(f"manifest task_id {row['task_id']} != requested index {task}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument(
        "--config", default="DSMC_0D_v2/config/full_domain_baseline_candidate.yaml")
    parser.add_argument("--artifact", required=True)
    parser.add_argument(
        "--reference-provenance",
        default="DSMC_0D_v2/reference/usf_reference_provenance_v1.json")
    args = parser.parse_args()

    row = manifest_row(Path(args.manifest), args.task)
    config = yaml.safe_load(Path(args.config).read_text())
    artifact = Path(args.artifact)
    provenance = Path(args.reference_provenance)
    if not artifact.is_file() or not provenance.is_file():
        raise FileNotFoundError("frozen artifact and reference provenance are required")
    artifact_hash = digest(artifact)
    if (row.get("protocol_version") != "usf-crossflow-v1"
            or row.get("artifact_sha256") != artifact_hash):
        raise ValueError("manifest is not bound to these frozen artifact bytes")

    alpha = float(row["alpha"])
    ar = float(row["aspect_ratio"])
    particles = int(row["particles"])
    phi = float(row["volume_fraction"])
    sphere = row["sphere"].lower() == "true"
    arm = row["arm"]
    expected_arm = "sphere_exact" if sphere else arm
    if expected_arm not in ("sphere_exact", "corrected", "uncorrected"):
        raise ValueError(f"unsupported USF study arm: {arm}")
    if sphere != (arm == "sphere_exact"):
        raise ValueError("sphere flag and arm disagree")
    if alpha >= 1.0:
        raise ValueError(
            "unthermostatted elastic USF has no steady state; alpha=1 is an HCS control")

    config["particle"]["AR"] = ar
    config["system"].update({
        "alpha": alpha,
        "kTt": float(row["initial_ttr"]),
        "kTr": float(row["initial_trot"]),
        "phi": phi,
        "particle_count": particles,
    })
    # Preserve volume fraction across shape and particle count.  The explicit
    # particle_count avoids the historical ceil/rounding ambiguity.
    params = particle_parameters(config)
    box = (particles * params.volume / phi) ** (1.0 / 3.0)
    config["system"]["domain"] = [box, box, box]
    if not math.isclose(particles * params.volume / box**3, phi,
                        rel_tol=2.0e-14, abs_tol=2.0e-14):
        raise RuntimeError("failed to construct the requested volume fraction")

    config["time"].update({
        "dt": float(row["dt"]),
        "dtau": 0.5,
        "t_end": 1.0e9,
        "tau_end": float(row["tau_end"]),
        "equilibration_time": 0.0,
    })
    config["flow"] = {"mode": "usf", "shear_rate": float(row["shear_rate"])}
    config["simulation"].update({
        "sphere_collision": sphere,
        "exact_initial_temperatures": True,
        "orientation_integrator": "symmetric_midpoint_v1",
        "use_isotropic_eps": True,
    })
    config["microscopic_closure"].update({
        "routing": "variational_v2",
        "angular": "variational_v2",
        "artifact": str(artifact),
        "invariant_corrections": arm == "corrected",
        "state_update_cpp": float(row["state_update_cpp"]),
        # Corrections remain local.  USF explicitly validates the effective
        # model that falls back to the unchanged base law outside that support.
        "correction_fallback_gate": "adaptive_base_law",
    })
    config.setdefault("diagnostics", {})["collision_audit"] = not sphere
    # The non-Gaussian accumulator remains off; its start value is also the
    # exact beginning of the closure fallback/stationarity evaluation window.
    config["diagnostics"]["non_gaussian"] = {
        "enabled": False,
        "sample_start_tau": float(row["evaluation_start_tau"]),
        "sample_end_tau": float(row["tau_end"]),
        "sample_delta_tau": 5.0,
    }

    prefix = Path(row["output_prefix"])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    trajectory = Path(str(prefix) + ".txt")
    pressure = Path(str(prefix) + "_pressure.txt")
    orientation = Path(str(prefix) + "_orientation.txt")
    diagnostics = run_simulation(
        config, int(row["seed"]), trajectory, pressure, orientation)
    diagnostics.update({
        "artifact": str(artifact),
        "artifact_sha256": artifact_hash,
        "artifact_manifest_sha256": digest(artifact.with_name("manifest.json")),
        "reference_provenance": str(provenance),
        "reference_provenance_sha256": digest(provenance),
        "usf_study_protocol": "usf-crossflow-v1",
        "usf_study_support_policy": "adaptive_base_law_v1",
        "validation_case": {
            "task_id": int(row["task_id"]),
            "source_task_id": int(row.get("source_task_id", row["task_id"])),
            "mode": row["mode"],
            "coordinate_role": row["coordinate_role"],
            "arm": arm,
            "alpha": alpha,
            "aspect_ratio": ar,
            "shear_rate": float(row["shear_rate"]),
            "rate_scale": float(row["rate_scale"]),
            "dt": float(row["dt"]),
            "initial_branch": row["initial_branch"],
            "replicate": int(row["replicate"]),
            "seed": int(row["seed"]),
            "particles": particles,
            "volume_fraction": phi,
            "particle_mass": float(params.mass),
            "particle_diameter": 2.0 * float(config["particle"]["radius"]),
            "box_length": box,
            "tau_end": float(row["tau_end"]),
            "evaluation_start_tau": float(row["evaluation_start_tau"]),
        },
    })
    destination = Path(str(prefix) + ".json")
    destination.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
