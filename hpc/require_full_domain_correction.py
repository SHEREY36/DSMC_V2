#!/usr/bin/env python3
"""Require one fitted correction node for every baseline artifact node."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def coordinate_set(values: np.ndarray) -> set[tuple[float, float, float]]:
    return {tuple(float(item) for item in row)
            for row in np.asarray(values, dtype=float)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()
    artifact = Path(args.artifact)
    if not artifact.is_file():
        raise SystemExit(f"missing artifact: {artifact}")
    with np.load(artifact, allow_pickle=False) as data:
        baseline = coordinate_set(data["surface_coordinates"])
        correction = coordinate_set(data["beta_coordinates"])
    missing = sorted(baseline - correction)
    extra = sorted(correction - baseline)
    if missing or extra:
        raise SystemExit(
            "artifact is not full-domain corrected: "
            f"baseline_nodes={len(baseline)}, correction_nodes={len(correction)}, "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}")
    print(f"full-domain correction coverage passes: {len(baseline)} nodes")


if __name__ == "__main__":
    main()
