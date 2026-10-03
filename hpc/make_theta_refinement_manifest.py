#!/usr/bin/env python3
"""CTC manifest for new temperature-ratio planes, theta = T_tr/T_rot.

The artifact had no node between theta = 0.2 and theta = 1 at any aspect ratio,
yet that gap holds the HCS attractors of AR 1.5 at every restitution, of AR 1.25,
and of AR 1.1 for alpha >= 0.9 (and of AR >= 2 at strong dissipation).  Each new
(theta, AR) coordinate gets one seed shared by all its alpha planes, as the
existing grid does, so differences between planes carry no sampling noise from
the incoming pairs.  Nodes already in the base grid are skipped.
"""
import argparse
import csv
from pathlib import Path

FIELDS = ["task_id", "stage", "role", "alpha", "theta", "aspect_ratio", "ensemble_id",
          "ensemble_mode", "control", "seed", "nsamples", "shard", "output_directory"]
ALPHAS = "0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-grid", required=True, help="current artifact grid manifest")
    parser.add_argument("--thetas", default="0.35,0.5,0.7")
    parser.add_argument("--alphas", default=ALPHAS)
    parser.add_argument("--aspect-ratios", default="1.1,1.2,1.35,1.5,2,2.5,3")
    parser.add_argument("--nsamples", type=int, default=200000)
    parser.add_argument("--seed-base", type=int, default=3610000)
    parser.add_argument("--results-root", default="results/ctc/nodes/training/theta_refinement")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = list(csv.DictReader(open(args.base_grid, newline="")))
    existing = {(round(float(r["alpha"]), 4), round(float(r["theta"]), 5),
                 round(float(r["aspect_ratio"]), 4)) for r in base}
    used_seeds = {int(r["seed"]) for r in base}
    rows = []
    thetas = [float(v) for v in args.thetas.split(",")]
    ars = [float(v) for v in args.aspect_ratios.split(",")]
    for i, theta in enumerate(thetas):
        for j, ar in enumerate(ars):
            seed = args.seed_base + 1000 * i + 10 * j
            if seed in used_seeds:
                raise SystemExit(f"seed {seed} already used by the base grid")
            for alpha in (float(v) for v in args.alphas.split(",")):
                if (round(alpha, 4), round(theta, 5), round(ar, 4)) in existing:
                    continue
                rows.append({
                    "task_id": len(rows), "stage": "theta_refinement", "role": "baseline",
                    "alpha": alpha, "theta": theta, "aspect_ratio": ar, "ensemble_id": 0,
                    "ensemble_mode": "baseline", "control": 0, "seed": seed,
                    "nsamples": args.nsamples, "shard": 0,
                    "output_directory": (f"{args.results_root}/alpha_{alpha:.3f}_theta_{theta:.3f}_"
                                         f"AR_{ar:.3f}_ensemble_000_shard_00")})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"{args.output}: {len(rows)} new nodes, theta {thetas} x AR {ars}")


if __name__ == "__main__":
    main()
