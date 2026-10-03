#!/usr/bin/env python3
"""Manifest for the encounter-unit USF campaign (usf-encounter-v1).

Rows are taken verbatim from the frozen usf-crossflow-v1 full manifest (same
seed, shear rate, dt, initial state, and tau window) for the rod aspect ratios
that have a DEM benchmark.  Only the collision model changes, through `arm`:

  encounter_memory  encounter clock + encounter loss + angular memory (proposed)
  encounter         encounter clock + encounter loss (ablation, replicate 0)
  contact_legacy    unchanged legacy runtime (same-code control, replicate 0)

Invariant corrections are off in every arm.  Rows are ordered longest first
so a capped Slurm array packs the tau=360 runs early.
"""
import argparse
import csv
from pathlib import Path

SOURCE = "manifests/usf_usf_crossflow_v1_20260927_full.csv"
AR_WITH_DEM = (1.5, 2.0, 2.5, 3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ablation-replicates", type=int, default=1,
                        help="replicates that also run the ablation and legacy arms; 0 = model only")
    args = parser.parse_args()
    with open(args.source, newline="") as handle:
        source = [row for row in csv.DictReader(handle)
                  if row["sphere"].lower() == "false"
                  and float(row["rate_scale"]) == 1.0
                  and any(abs(float(row["aspect_ratio"]) - ar) < 1e-9 for ar in AR_WITH_DEM)]
    rows = []
    for row in source:
        replicate = int(row["replicate"])
        arms = ["encounter_memory"]
        if replicate < args.ablation_replicates:
            arms.append("encounter")
            # Same-code legacy control where the legacy error is largest.
            if float(row["aspect_ratio"]) >= 2.0 - 1e-9 and float(row["aspect_ratio"]) != 2.5:
                arms.append("contact_legacy")
        for arm in arms:
            prefix = (f"results/validation/usf/usf_encounter_{args.tag}/{arm}/"
                      f"AR_{float(row['aspect_ratio']):.3f}_alpha_{float(row['alpha']):.3f}_"
                      f"{row['initial_branch']}_rep_{replicate:02d}")
            rows.append({
                "arm": arm, "alpha": row["alpha"], "aspect_ratio": row["aspect_ratio"],
                "shear_rate": row["shear_rate"], "dt": row["dt"],
                "initial_branch": row["initial_branch"], "initial_ttr": row["initial_ttr"],
                "initial_trot": row["initial_trot"], "replicate": replicate,
                "seed": row["seed"], "particles": row["particles"],
                "volume_fraction": row["volume_fraction"], "tau_end": row["tau_end"],
                "evaluation_start_tau": row["evaluation_start_tau"],
                "state_update_cpp": row["state_update_cpp"], "output_prefix": prefix,
                "protocol_version": "usf-encounter-v1"})
    rows.sort(key=lambda r: (-float(r["tau_end"]), r["arm"], float(r["aspect_ratio"]),
                             float(r["alpha"]), r["initial_branch"], r["replicate"]))
    for index, row in enumerate(rows):
        row["task_id"] = index
    fields = ["task_id"] + [key for key in rows[0] if key != "task_id"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["arm"]] = counts.get(row["arm"], 0) + 1
    print(f"{args.output}: {len(rows)} rows {counts}")


if __name__ == "__main__":
    main()
