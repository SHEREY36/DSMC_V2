#!/usr/bin/env python3
"""Generate pilot, production, or QA-driven continuation manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ALPHAS = [round(0.50 + 0.05 * i, 2) for i in range(10)] + [0.975, 0.99]
THETAS = [round(0.1 * i, 1) for i in range(1, 13)]
ASPECT_RATIOS = [1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
FIELDS = ["task_id", "stage", "role", "alpha", "theta", "aspect_ratio",
          "seed", "nsamples", "shard", "output_directory"]


def seed_for(theta_index: int, ar_index: int, shard: int) -> int:
    # Alpha is intentionally absent: a fixed (theta,AR,shard) uses common draws.
    return 20260826 + 100000 * ar_index + 1000 * theta_index + shard


def base_rows(stage: str, samples: int, shard: int, results_root: str):
    rows = []
    task = 0
    for ai, ar in enumerate(ASPECT_RATIOS):
        for ti, theta in enumerate(THETAS):
            for alpha in ALPHAS:
                rows.append({"task_id": task, "stage": stage, "role": "routing",
                    "alpha": alpha, "theta": theta, "aspect_ratio": ar,
                    "seed": seed_for(ti, ai, shard), "nsamples": samples, "shard": shard,
                    "output_directory": f"{results_root}/{stage}/alpha_{alpha:.3f}_theta_{theta:.3f}_AR_{ar:.3f}_shard_{shard:02d}"})
                task += 1
        rows.append({"task_id": task, "stage": stage, "role": "vss_elastic_reference",
            "alpha": 1.0, "theta": 1.0, "aspect_ratio": ar,
            "seed": seed_for(9, ai, shard), "nsamples": samples, "shard": shard,
            "output_directory": f"{results_root}/{stage}/alpha_1.000_theta_1.000_AR_{ar:.3f}_shard_{shard:02d}"})
        task += 1
    return rows


def continuation_rows(qa_path: str, samples: int, shard: int, results_root: str):
    rows = []
    with open(qa_path, newline="") as handle:
        for source in csv.DictReader(handle):
            if source["precision_pass"].lower() == "true":
                continue
            if int(source["n_outcomes"]) >= 1000000:
                continue
            alpha, theta, ar = map(float, (source["alpha"], source["theta"], source["aspect_ratio"]))
            ai = ASPECT_RATIOS.index(ar)
            ti = min(range(len(THETAS)), key=lambda i: abs(THETAS[i] - theta))
            role = "vss_elastic_reference" if alpha >= 1.0 else "routing"
            rows.append({"task_id": len(rows), "stage": "continuation", "role": role,
                "alpha": alpha, "theta": theta, "aspect_ratio": ar,
                "seed": seed_for(ti, ai, shard), "nsamples": samples, "shard": shard,
                "output_directory": f"{results_root}/continuation/alpha_{alpha:.3f}_theta_{theta:.3f}_AR_{ar:.3f}_shard_{shard:02d}"})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "production", "continuation"), required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--qa-summary")
    parser.add_argument("--results-root", default="results/ctc")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    defaults = {"pilot": (20000, 0), "production": (80000, 1), "continuation": (100000, 2)}
    samples, shard = args.samples or defaults[args.stage][0], args.shard if args.shard is not None else defaults[args.stage][1]
    if args.stage == "continuation":
        if not args.qa_summary:
            parser.error("continuation requires --qa-summary")
        rows = continuation_rows(args.qa_summary, samples, shard, args.results_root)
    else:
        rows = base_rows(args.stage, samples, shard, args.results_root)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        # The rows are consumed by Bash as well as Python.  Force Unix newlines
        # so the final output_directory field never acquires a literal '\r'.
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tasks to {path}")


if __name__ == "__main__":
    main()
