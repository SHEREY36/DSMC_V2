#!/usr/bin/env python3
"""CTC manifest for new restitution planes of the closure grid.

Every (theta, AR) coordinate of the existing canonical grid receives a node at
each requested alpha that is not already present.  A new node uses the seed of
the alpha = 0.5 node at the same (theta, AR): the generator's attempt stream is
common across alpha, so the new planes see exactly the incoming pairs of the
existing planes and differences along alpha carry no sampling noise from the
proposals.  Rows use the 13-field closure format read by hpc/run_ctc_row.sh.
"""
import argparse
import csv
from pathlib import Path

FIELDS = ["task_id", "stage", "role", "alpha", "theta", "aspect_ratio", "ensemble_id",
          "ensemble_mode", "control", "seed", "nsamples", "shard", "output_directory"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-grid", default="manifests/artifact_grid.csv")
    parser.add_argument("--alphas", default="0.55,0.6,0.65,0.7,0.75,0.85,0.9")
    parser.add_argument("--nsamples", type=int, default=200000)
    parser.add_argument("--results-root", default="results/ctc_closure_200k/alpha_refinement")
    parser.add_argument("--aspect-ratios", default=None, help="optional comma list filter")
    parser.add_argument("--thetas", default=None, help="optional comma list filter")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    alphas = [round(float(a), 4) for a in args.alphas.split(",")]
    keep_ar = None if not args.aspect_ratios else {round(float(v), 4) for v in args.aspect_ratios.split(",")}
    keep_th = None if not args.thetas else {round(float(v), 5) for v in args.thetas.split(",")}
    with open(args.base_grid, newline="") as handle:
        base = list(csv.DictReader(handle))
    existing = {(round(float(r["alpha"]), 4), round(float(r["theta"]), 5),
                 round(float(r["aspect_ratio"]), 4)) for r in base}
    seed_at = {}
    for r in base:
        key = (round(float(r["theta"]), 5), round(float(r["aspect_ratio"]), 4))
        if abs(float(r["alpha"]) - 0.5) < 1e-9 and int(r.get("ensemble_id", 0)) == 0:
            seed_at[key] = int(r["seed"])
    coordinates = sorted(seed_at)
    rows = []
    for theta, ar in coordinates:
        if keep_ar is not None and ar not in keep_ar:
            continue
        if keep_th is not None and theta not in keep_th:
            continue
        for alpha in alphas:
            if (alpha, theta, ar) in existing:
                continue
            rows.append({
                "task_id": len(rows), "stage": "alpha_refinement", "role": "baseline",
                "alpha": alpha, "theta": theta, "aspect_ratio": ar, "ensemble_id": 0,
                "ensemble_mode": "baseline", "control": 0, "seed": seed_at[(theta, ar)],
                "nsamples": args.nsamples, "shard": 0,
                "output_directory": (f"{args.results_root}/alpha_{alpha:.3f}_theta_{theta:.3f}_"
                                     f"AR_{ar:.3f}_ensemble_000_shard_00")})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.output}: {len(rows)} new nodes over {len(coordinates)} (theta, AR) "
          f"coordinates at alpha {alphas}")


if __name__ == "__main__":
    main()
