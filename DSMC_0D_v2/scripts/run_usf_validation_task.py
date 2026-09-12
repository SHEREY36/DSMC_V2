#!/usr/bin/env python3
"""Run one frozen-artifact uniform-shear validation realization."""

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


def sha256(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def row_at(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= index < len(rows):
        raise IndexError(f"task {index} outside manifest with {len(rows)} rows")
    return rows[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--config", default="DSMC_0D_v2/config/default.yaml")
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()

    row = row_at(Path(args.manifest), args.task)
    config = yaml.safe_load(Path(args.config).read_text())
    alpha = float(row["alpha"])
    ar = float(row["aspect_ratio"])
    particles = int(row["particles"])
    arm = row["arm"]
    if arm not in ("corrected", "uncorrected"):
        raise ValueError(f"unsupported validation arm: {arm}")

    config["particle"]["AR"] = ar
    config["system"].update({"alpha": alpha, "kTt": 1.0, "kTr": 1.0})
    # Keep number density and N exact across aspect ratios.  With 4000
    # particles in this box n=0.01526, close to the archived dilute-USF data.
    box = 64.0
    config["system"]["domain"] = [box, box, box]
    params = particle_parameters(config)
    config["system"]["phi"] = (
        (particles - 0.25) * params.volume / box**3)
    derived = math.ceil(config["system"]["phi"] * box**3 / params.volume)
    if derived != particles:
        raise RuntimeError(f"requested {particles} particles, derived {derived}")
    config["time"].update({
        "dt": 0.01,
        "dtau": 0.2,
        "t_end": 100000.0,
        "tau_end": float(row["tau_end"]),
        "equilibration_time": 0.0,
    })
    config["flow"] = {
        "mode": "usf", "shear_rate": float(row["shear_rate"])}
    config["microscopic_closure"].update({
        "routing": "variational_v2",
        "angular": "variational_v2",
        "artifact": args.artifact,
        "invariant_corrections": arm == "corrected",
    })
    config.setdefault("diagnostics", {})["collision_audit"] = True

    prefix = Path(row["output_prefix"])
    trajectory = Path(str(prefix) + ".txt")
    pressure = Path(str(prefix) + "_pressure.txt")
    orientation = Path(str(prefix) + "_orientation.txt")
    diagnostics = run_simulation(
        config, int(row["seed"]), trajectory, pressure, orientation)
    diagnostics["artifact"] = str(Path(args.artifact))
    diagnostics["artifact_sha256"] = sha256(args.artifact)
    diagnostics["validation_case"] = {
        "mode": row["mode"],
        "arm": arm,
        "alpha": alpha,
        "aspect_ratio": ar,
        "shear_rate": float(row["shear_rate"]),
        "replicate": int(row["replicate"]),
        "seed": int(row["seed"]),
        "particles": particles,
        "tau_end": float(row["tau_end"]),
    }
    Path(str(prefix) + ".json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
