#!/usr/bin/env python3
"""Create staged, one-realization-per-task non-Gaussian HCS designs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FIELDS = (
    "task_id", "mode", "arm", "alpha", "aspect_ratio", "replicate",
    "seed", "particles", "tau_end", "sample_start_tau", "sample_end_tau",
    "sample_delta_tau", "output_prefix",
)
SEEDS = (260916101, 260916211, 260916307, 260916419, 260916523,
         260916631, 260916733, 260916839, 260916947, 260917051,
         260917159, 260917267, 260917373, 260917481, 260917589,
         260917697)


def surface_axes(artifact: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    with np.load(artifact, allow_pickle=False) as data:
        coordinates = np.asarray(data["surface_coordinates"], dtype=float)
    return (tuple(np.unique(coordinates[:, 0]).tolist()),
            tuple(np.unique(coordinates[:, 2]).tolist()))


def design(mode: str, artifact: Path):
    if mode == "engineering":
        cases = ((0.80, 2.0), (0.95, 2.0), (0.80, 3.0))
        return cases, SEEDS[:4], ("scaled", "unscaled"), 2000, 40.0, 20.0, 2.0
    if mode == "domain-pilot":
        alphas, ars = surface_axes(artifact)
        return tuple((a, ar) for a in alphas for ar in ars), SEEDS[:4], \
            ("scaled",), 4000, 200.0, 100.0, 5.0
    if mode == "map":
        _, ars = surface_axes(artifact)
        alphas = tuple(np.round(np.arange(0.50, 1.00, 0.05), 2))
        return tuple((a, ar) for a in alphas for ar in ars), SEEDS, \
            ("scaled",), 10000, 1500.0, 500.0, 5.0
    if mode == "tails":
        cases = tuple((a, ar) for a in (0.50, 0.75, 0.95)
                      for ar in (1.10, 2.00, 3.00))
        # One hundred independent realizations are generated deterministically
        # below; the first sixteen are shared with the map campaign.
        seeds = tuple(260916101 + 1009 * index for index in range(100))
        return cases, seeds, ("scaled",), 10000, 1500.0, 500.0, 5.0
    if mode == "sphere-controls":
        cases = tuple((a, 1.0) for a in (0.50, 0.75, 0.95, 1.00))
        return cases, SEEDS, ("sphere",), 10000, 1500.0, 500.0, 5.0
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("engineering", "domain-pilot", "map",
                                            "tails", "sphere-controls"),
                        default="engineering")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    artifact = Path(args.artifact)
    if not artifact.is_file():
        raise SystemExit(f"missing artifact: {artifact}")
    cases, seeds, arms, particles, tau_end, start, delta = design(args.mode, artifact)
    rows = []
    for alpha, ar in cases:
        for arm in arms:
            for replicate, seed in enumerate(seeds):
                tag = (f"alpha_{alpha:.2f}_AR_{ar:.2f}_{arm}_"
                       f"rep_{replicate:03d}")
                rows.append({
                    "task_id": len(rows), "mode": args.mode, "arm": arm,
                    "alpha": f"{alpha:.2f}", "aspect_ratio": f"{ar:.2f}",
                    "replicate": replicate, "seed": seed,
                    "particles": particles, "tau_end": tau_end,
                    "sample_start_tau": start, "sample_end_tau": tau_end,
                    "sample_delta_tau": delta,
                    "output_prefix": str(Path(args.results) / tag),
                })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} {args.mode} non-Gaussian HCS tasks to {output}")


if __name__ == "__main__":
    main()
