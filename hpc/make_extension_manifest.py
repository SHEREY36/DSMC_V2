#!/usr/bin/env python3
"""Manifests for the two aspect-ratio extensions.

ar_extension
    AR = 1.5 and 2.5 on the same (alpha, theta) grid the sentinel already
    covers, so the closure is fitted on five aspect ratios rather than three
    and the surface is interpolated rather than extrapolated between them.

ar_low_theta
    AR = 1.1 at theta = 0.025, 0.05, 0.1. The sentinel grid stops at theta =
    0.2, and at AR = 1.1 the inelastic fixed point sits below that: the fits
    clamp at the grid edge and the stability gate refuses the node, correctly.
    Rubio-Largo et al. (Physica A 443, 2016, Fig. 2b) put the steady ratio at
    0.31 to 0.60 for e_n = 0.88 to 0.96 at this aspect ratio, so the attractor
    is real and simply outside the window we sampled.

Seeds are offset by aspect-ratio index so no row can collide with the sentinel
campaign, which shares the generator's seed function.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = ("task_id", "stage", "role", "alpha", "theta", "aspect_ratio",
          "ensemble_id", "ensemble_mode", "control", "seed", "nsamples",
          "shard", "output_directory")

ALPHA = (0.5, 0.8, 0.95, 1.0)
DESIGNS = {
    # aspect ratios that fill the gap between the fitted ones
    "ar_extension": {"ar": (1.5, 2.5), "theta": (0.2, 1.0, 2.0), "ar_offset": 10},
    # The near-sphere boundary layer. Exchange switches off as the torque arm
    # vanishes, so resolving it needs points close together in AR, not a single
    # value at 1.1. These also let the AR scaling of the exchange strength be
    # FITTED rather than assumed: on the three aspect ratios available now,
    # p_exch/(AR-1)^2 runs 5.55, 0.71, 0.19, so the naive quadratic torque law
    # is not what the data shows.
    "ar_near_sphere": {"ar": (1.2, 1.35), "theta": (0.2, 1.0, 2.0), "ar_offset": 30},
    # theta below the sentinel floor, where AR = 1.1 actually settles
    "ar_low_theta": {"ar": (1.1,), "theta": (0.0125, 0.025, 0.05, 0.1, 0.15),
                     "ar_offset": 20},
}


def rows_for(design: str, samples: int, root: str) -> list[dict]:
    spec = DESIGNS[design]
    rows = []
    for ri, ar in enumerate(spec["ar"]):
        for ti, theta in enumerate(spec["theta"]):
            for ai, alpha in enumerate(ALPHA):
                tag = (f"alpha_{alpha:.3f}_theta_{theta:.3f}_AR_{ar:.3f}"
                       f"_ensemble_000_shard_00")
                seed = (260902 + 100000 * (ri + spec["ar_offset"])
                        + 5000 * ti + ai)
                rows.append({
                    "task_id": len(rows), "stage": design, "role": "baseline",
                    "alpha": alpha, "theta": theta, "aspect_ratio": ar,
                    "ensemble_id": 0, "ensemble_mode": "baseline", "control": "0",
                    "seed": seed, "nsamples": samples, "shard": 0,
                    "output_directory": f"{root}/{design}/{tag}",
                })
    # Every fit resolves its reference law from the elastic, equipartitioned
    # shard sitting beside it, so a design that does not already contain
    # (alpha = 1, theta = 1) has to carry one or nothing in it can be anchored.
    for ri, ar in enumerate(spec["ar"]):
        if any(abs(r["alpha"] - 1.0) < 1e-9 and abs(r["theta"] - 1.0) < 1e-9
               and abs(r["aspect_ratio"] - ar) < 1e-9 for r in rows):
            continue
        tag = (f"alpha_1.000_theta_1.000_AR_{ar:.3f}_ensemble_000_shard_00")
        rows.append({
            "task_id": len(rows), "stage": design, "role": "anchor",
            "alpha": 1.0, "theta": 1.0, "aspect_ratio": ar,
            "ensemble_id": 0, "ensemble_mode": "baseline", "control": "0",
            "seed": 260902 + 100000 * (ri + spec["ar_offset"]) + 4999,
            "nsamples": samples, "shard": 0,
            "output_directory": f"{root}/{design}/{tag}",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=sorted(DESIGNS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nsamples", type=int, default=200000)
    parser.add_argument("--root", default="results/ctc_closure_200k")
    args = parser.parse_args()

    rows = rows_for(args.design, args.nsamples, args.root)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    seeds = {row["seed"] for row in rows}
    assert len(seeds) == len(rows), "seed collision within the design"
    print(f"{args.output}: {len(rows)} rows, "
          f"AR {sorted({r['aspect_ratio'] for r in rows})}, "
          f"theta {sorted({r['theta'] for r in rows})}")


if __name__ == "__main__":
    main()
