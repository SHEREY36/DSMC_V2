#!/usr/bin/env python3
"""Create paired pilot or corrected-only production USF validation rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SHEAR_RATES = {
    0.50: 0.0800,
    0.55: 0.0771,
    0.60: 0.0739,
    0.65: 0.0702,
    0.70: 0.0660,
    0.75: 0.0611,
    0.80: 0.0554,
    0.85: 0.0487,
    0.90: 0.0403,
    0.95: 0.0288,
}


def campaign_rows(mode: str, results: Path, particles: int,
                  tau_end: float) -> list[dict[str, object]]:
    """Return independent one-core jobs without changing any discretization."""
    if mode == "pilot":
        aspect_ratios = (2.0,)
        alphas = (0.50, 0.75, 0.95)
        arms = ("uncorrected", "corrected")
    elif mode == "full":
        aspect_ratios = (1.5, 2.0, 2.5, 3.0)
        alphas = tuple(sorted(SHEAR_RATES))
        arms = ("corrected",)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    seeds = (260913101, 260913211, 260913307, 260913419)
    rows: list[dict[str, object]] = []
    for ar in aspect_ratios:
        for alpha in alphas:
            for arm in arms:
                for replicate, seed in enumerate(seeds):
                    prefix = results / (
                        f"AR_{ar:.2f}_alpha_{alpha:.2f}_{arm}_rep_{replicate:02d}")
                    rows.append({
                        "task_id": len(rows),
                        "mode": mode,
                        "arm": arm,
                        "alpha": f"{alpha:.2f}",
                        "aspect_ratio": f"{ar:.2f}",
                        "shear_rate": f"{SHEAR_RATES[alpha]:.4f}",
                        "replicate": replicate,
                        "seed": seed,
                        "particles": int(particles),
                        "tau_end": f"{float(tau_end):.1f}",
                        "output_prefix": str(prefix),
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output", default="manifests/usf_validation_pilot.csv")
    parser.add_argument("--results", default="results/usf_validation_pilot")
    parser.add_argument("--particles", type=int, default=4000)
    parser.add_argument("--tau-end", type=float, default=80.0)
    args = parser.parse_args()
    if args.particles < 100:
        raise SystemExit("USF validation requires at least 100 particles")
    if args.tau_end <= 0.0:
        raise SystemExit("--tau-end must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = campaign_rows(args.mode, Path(args.results), args.particles,
                         args.tau_end)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} {args.mode} USF tasks to {output}")


if __name__ == "__main__":
    main()
