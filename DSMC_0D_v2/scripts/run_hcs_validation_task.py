#!/usr/bin/env python3
"""Run one HCS attractor-validation row without creating temporary YAML."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from dsmc_v2.particle import particle_parameters
from dsmc_v2.simulation import run_simulation


def row_at(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= index < len(rows):
        raise IndexError(f"task {index} outside manifest with {len(rows)} rows")
    return rows[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/hcs_validation.csv")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--config", default="DSMC_0D_v2/config/default.yaml")
    parser.add_argument("--artifact", default="models/microscopic_closure_v2/closure_v2.npz")
    args = parser.parse_args()

    row = row_at(Path(args.manifest), args.task)
    config = yaml.safe_load(Path(args.config).read_text())
    alpha = float(row["alpha"])
    ar = float(row["aspect_ratio"])
    theta0 = float(row["theta0"])
    particles = int(row["particles"])

    config["particle"]["AR"] = ar
    config["system"]["alpha"] = alpha
    # Keep initial total modal energy fixed while changing the partition:
    # (3/2) T_tr + T_rot = 5/2 at equipartition units.
    ktr = 2.5 / (1.0 + 1.5 * theta0)
    config["system"]["kTr"] = ktr
    config["system"]["kTt"] = theta0 * ktr
    config["system"]["domain"] = [64.0, 64.0, 64.0]
    params = particle_parameters(config)
    # The simulator derives N with ceil(phi*V/vp).  A quarter-particle inward
    # offset avoids floating-point roundoff turning an intended 2003 into 2004.
    config["system"]["phi"] = (particles - 0.25) * params.volume / (64.0 ** 3)
    derived_particles = math.ceil(config["system"]["phi"] * 64.0 ** 3 / params.volume)
    if derived_particles != particles:
        raise RuntimeError(f"requested {particles} particles, derived {derived_particles}")
    config["time"].update({"dt": 0.01, "dtau": 0.1,
                            "t_end": 10000.0, "tau_end": float(row["tau_end"]),
                            "equilibration_time": 0.0})
    config["flow"] = {"mode": "hcs", "shear_rate": 0.0}
    config["microscopic_closure"].update({
        "routing": "variational_v2", "angular": "variational_v2",
        "artifact": args.artifact, "invariant_corrections": False,
    })
    config.setdefault("diagnostics", {})["collision_audit"] = True

    prefix = Path(row["output_prefix"])
    trajectory_path = Path(str(prefix) + ".txt")
    diagnostics = run_simulation(config, int(row["seed"]), trajectory_path)
    diagnostics["validation_case"] = {
        "tier": row["tier"], "alpha": alpha, "aspect_ratio": ar,
        "theta0": theta0,
        "replicate": int(row.get("replicate", 0)),
        "target_theta": None if not row["target_theta"] else float(row["target_theta"]),
    }
    path = Path(str(prefix) + ".json")
    path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
