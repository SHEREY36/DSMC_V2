#!/usr/bin/env python3
"""Assemble the alpha-refined closure grid and the list of nodes still to fit.

* grid manifest: the existing canonical grid (repairs first, so their kernel
  forms win) plus the new alpha planes, one shared elastic anchor per AR;
* estimates directory: existing current-contract node estimates are reused;
* fit manifest: the grid rows that have no estimate yet (the new planes).
Runs before the CTC finishes: nothing here reads a shard.
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

BASE = ["manifests/artifact_grid_repairs.csv", "manifests/closure_sentinel.csv",
        "manifests/ar_extension.csv", "manifests/ar_near_sphere.csv",
        "manifests/ar_low_theta.csv"]


def estimate_name(row: dict) -> str:
    return (f"alpha_{float(row['alpha']):.3f}_theta_{float(row['theta']):.3f}_"
            f"AR_{float(row['aspect_ratio']):.3f}_ensemble_{int(row.get('ensemble_id', 0)):03d}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-manifest", required=True)
    parser.add_argument("--old-estimates", default="results/closure_estimates/artifact_grid")
    parser.add_argument("--estimates", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--fit", required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "hpc/make_artifact_manifest.py", *BASE,
                    args.refinement_manifest, "--output", args.grid], check=True)
    with open(args.grid, newline="") as handle:
        grid = list(csv.DictReader(handle))
    estimates = Path(args.estimates)
    estimates.mkdir(parents=True, exist_ok=True)
    reused, to_fit = 0, []
    for row in grid:
        name = estimate_name(row)
        target = estimates / name
        source = Path(args.old_estimates) / name
        if not target.is_file() and source.is_file():
            shutil.copy2(source, target)
        if target.is_file():
            reused += 1
        else:
            to_fit.append(dict(row))
    for index, row in enumerate(to_fit):
        row["task_id"] = str(index)
    fields = list(grid[0].keys())
    with open(args.fit, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(to_fit)
    print(f"grid {args.grid}: {len(grid)} nodes; reused {reused} estimates; "
          f"{len(to_fit)} nodes to fit -> {args.fit}")


if __name__ == "__main__":
    main()
