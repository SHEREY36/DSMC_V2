#!/usr/bin/env python3
"""Populate one unique baseline shard's deterministic propensity cache."""

from __future__ import annotations

import argparse
import csv

import numpy as np

from coll_models_v2.estimate import _run_propensity
from dsmc_v2_contracts import load_run, validate_run


def unique_baselines(manifest: str) -> list[str]:
    with open(manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "baseline_directory" not in rows[0]:
        raise ValueError("excitation manifest has no baseline_directory rows")
    return sorted({row["baseline_directory"] for row in rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--offsets", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    baselines = unique_baselines(args.manifest)
    if not 0 <= args.index < len(baselines):
        raise IndexError(
            f"propensity task {args.index} outside 0..{len(baselines) - 1}")
    run = load_run(baselines[args.index])
    qa = validate_run(run)
    if qa["status"] != "pass":
        raise ValueError(f"baseline CTC shard failed validation: {qa}")
    values = _run_propensity(run, args.offsets, workers=args.workers)
    if values is None or len(values) != len(run.attempts) \
            or not np.all(np.isfinite(values)):
        raise ValueError("propensity cache result is incomplete or non-finite")
    print(
        f"propensity node {args.index + 1}/{len(baselines)}: "
        f"{baselines[args.index]} ({len(values)} attempts, "
        f"{args.offsets} offsets, {args.workers} workers)",
        flush=True,
    )


if __name__ == "__main__":
    main()
