#!/usr/bin/env python3
"""Create the fresh target/anchor CTC pair for the response holdout gate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CTC_FIELDS = (
    "task_id", "stage", "role", "alpha", "theta", "aspect_ratio",
    "ensemble_id", "ensemble_mode", "control", "seed", "nsamples",
    "shard", "output_directory",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctc-output", required=True)
    parser.add_argument("--fit-output", required=True)
    parser.add_argument("--results-root",
                        default="results/ctc_independent_holdout_v1")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=260_912_001)
    args = parser.parse_args()
    if not 1 <= args.samples <= 200_000:
        parser.error("--samples must lie in 1..200000")
    root = Path(args.results_root)
    target = {
        "task_id": 0, "stage": "independent_holdout",
        "role": "fresh_response_target", "alpha": 0.95, "theta": 1.0,
        "aspect_ratio": 2.0, "ensemble_id": 0,
        "ensemble_mode": "baseline", "control": "0",
        "seed": args.seed, "nsamples": args.samples, "shard": 0,
        "output_directory": str(
            root / "alpha_0.950_theta_1.000_AR_2.000_ensemble_000_shard_00"),
    }
    anchor = {
        "task_id": 1, "stage": "independent_holdout",
        "role": "fresh_equilibrium_anchor", "alpha": 1.0, "theta": 1.0,
        "aspect_ratio": 2.0, "ensemble_id": 0,
        "ensemble_mode": "baseline", "control": "0",
        # Preserve the CTC common-random-number contract along the alpha line.
        # This stream is new relative to training, but shared between target
        # and anchor so their finite-sample difference is not inflated.
        "seed": args.seed, "nsamples": args.samples, "shard": 0,
        "output_directory": str(
            root / "alpha_1.000_theta_1.000_AR_2.000_ensemble_000_shard_00"),
    }

    ctc_output = Path(args.ctc_output)
    ctc_output.parent.mkdir(parents=True, exist_ok=True)
    with ctc_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CTC_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows((target, anchor))

    fit_output = Path(args.fit_output)
    fit_output.parent.mkdir(parents=True, exist_ok=True)
    fit_fields = CTC_FIELDS + ("anchor_directory",)
    with fit_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fit_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({**target, "anchor_directory": anchor["output_directory"]})
    print(f"wrote two fresh CTC rows to {ctc_output} and one fit row to {fit_output}")


if __name__ == "__main__":
    main()
