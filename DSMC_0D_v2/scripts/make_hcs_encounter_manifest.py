#!/usr/bin/env python3
"""HCS re-validation of the encounter-unit closure at the DEM benchmark nodes.

The encounter unit moves the HCS attractor only through the per-encounter loss
(the clock rescales time, which theta* is invariant to; the angular law does not
act on a homogeneous isotropic state).  Two starts bracket each attractor.
"""
import argparse
import csv
from pathlib import Path

# DEM HCS temperature ratios (T_tr/T_rot), two-seed means of the corrected DEM in
# LAMMPS/runs/fresh_hcs/analysis/summary.csv (exact Hertz damping).  The earlier
# targets 0.9365 / 1.0158 at alpha = 0.8 came from runs/HCS, which predates the
# damping fix and is not used.
GATE = ((1.00, 2.0, 1.0000), (1.00, 3.0, 1.0000), (0.95, 2.0, 0.9790),
        (0.95, 3.0, 0.9960), (0.80, 2.0, 0.9586), (0.80, 3.0, 1.0049))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--particles", type=int, default=10000)
    parser.add_argument("--tau-end", type=float, default=80.0)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--arms", default="encounter_memory,contact_legacy")
    args = parser.parse_args()
    rows = []
    for arm in args.arms.split(","):
        for alpha, ar, target in GATE:
            for theta0 in (0.5, 2.0):
                for rep in range(args.replicates):
                    rows.append({
                        "task_id": len(rows), "tier": "gate", "arm": arm, "alpha": alpha,
                        "aspect_ratio": ar, "theta0": theta0, "target_theta": target,
                        "replicate": rep, "seed": 290930 + 97 * len(rows),
                        "particles": args.particles, "tau_end": args.tau_end,
                        "state_update_cpp": 0.05,
                        "output_prefix": (f"results/validation/hcs/hcs_encounter_{args.tag}/{arm}/alpha_{alpha:.2f}_"
                                          f"AR_{ar:.2f}_theta0_{theta0:.2f}_rep_{rep:02d}")})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"{args.output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
