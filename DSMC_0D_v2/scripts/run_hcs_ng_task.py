#!/usr/bin/env python3
"""Execute one frozen-artifact non-Gaussian HCS realization."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_at(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= index < len(rows):
        raise IndexError(f"task {index} outside manifest with {len(rows)} rows")
    return rows[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True, type=int)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--config", default="DSMC_0D_v2/config/default.yaml")
    args = parser.parse_args()
    row = row_at(Path(args.manifest), args.task)
    artifact = Path(args.artifact)
    config = yaml.safe_load(Path(args.config).read_text())
    alpha, ar = float(row["alpha"]), float(row["aspect_ratio"])
    sphere = row["arm"] == "sphere"
    config["particle"]["AR"] = ar
    config["system"].update(alpha=alpha, kTt=1.0, kTr=1.0,
                            domain=[64.0, 64.0, 64.0])
    config.setdefault("simulation", {})["sphere_collision"] = sphere
    config["simulation"]["hcs_rescale_temperature"] = row["arm"] == "scaled"
    if sphere:
        config["microscopic_closure"].update(
            routing="legacy_rank0", angular="legacy",
            invariant_corrections=False)
    else:
        config["microscopic_closure"].update(
            routing="variational_v2", angular="variational_v2",
            artifact=str(artifact), invariant_corrections=True)
    params = particle_parameters(config)
    particles = int(row["particles"])
    volume = 64.0**3
    config["system"]["phi"] = (particles - 0.25) * params.volume / volume
    if math.ceil(config["system"]["phi"] * volume / params.volume) != particles:
        raise RuntimeError("failed to derive requested particle count")
    delta = float(row["sample_delta_tau"])
    config["time"].update(dt=0.01, dtau=min(1.0, delta), t_end=100000.0,
                          tau_end=float(row["tau_end"]), equilibration_time=0.0)
    config["flow"] = {"mode": "hcs", "shear_rate": 0.0}
    config.setdefault("diagnostics", {})["collision_audit"] = True
    config["diagnostics"]["non_gaussian"] = {
        "enabled": True,
        "sample_start_tau": float(row["sample_start_tau"]),
        "sample_end_tau": float(row["sample_end_tau"]),
        "sample_delta_tau": delta,
        "minimum_tail_count": 1000,
    }
    prefix = Path(row["output_prefix"])
    trajectory = Path(str(prefix) + ".txt")
    result = run_simulation(config, int(row["seed"]), trajectory)
    result["artifact"] = str(artifact)
    result["artifact_sha256"] = sha256(artifact)
    result["campaign"] = {
        key: (float(row[key]) if key in ("alpha", "aspect_ratio")
              else int(row[key]) if key in ("replicate", "seed", "particles")
              else row[key])
        for key in ("mode", "arm", "alpha", "aspect_ratio", "replicate",
                    "seed", "particles")
    }
    output = Path(str(prefix) + ".json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
