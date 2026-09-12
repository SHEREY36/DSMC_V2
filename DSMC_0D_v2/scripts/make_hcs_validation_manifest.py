#!/usr/bin/env python3
"""Create the compact HCS gate that precedes excitation/USF validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "task_id", "tier", "alpha", "aspect_ratio", "theta0", "target_theta",
    "replicate", "seed", "particles", "tau_end", "output_prefix",
)

# DEM HCS ratios used in the coupling handoff.  The elastic entries are exact
# physics rulers, not fitted targets.
GATE_CASES = (
    ("gate", 1.00, 2.0, 1.0000),
    ("gate", 1.00, 3.0, 1.0000),
    ("gate", 0.95, 2.0, 0.9792),
    ("gate", 0.95, 3.0, 0.9962),
    ("gate", 0.80, 2.0, 0.9365),
    ("gate", 0.80, 3.0, 1.0158),
)

# Transfer/interpolation diagnostics do not block the known DEM cases because
# no independent DEM target is encoded here.  Submit them only after the fast
# truth-backed gate, rather than spending compute before it changes a decision.
DIAGNOSTIC_CASES = (
    ("diagnostic", 0.95, 1.5, None),
    ("diagnostic", 0.95, 2.5, None),
    ("diagnostic", 0.80, 1.1, None),
    ("diagnostic", 0.80, 1.2, None),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="manifests/hcs_validation.csv")
    parser.add_argument("--results", default="results/hcs_validation")
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--tau-end", type=float, default=20.0)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--include-diagnostics", action="store_true")
    args = parser.parse_args()

    if args.replicates < 1:
        parser.error("--replicates must be positive")
    rows = []
    cases = GATE_CASES + (DIAGNOSTIC_CASES if args.include_diagnostics else ())
    for case_index, (tier, alpha, ar, target) in enumerate(cases):
        for start_index, theta0 in enumerate((0.75, 1.25)):
            for replicate in range(args.replicates):
                tag = (f"alpha_{alpha:.2f}_AR_{ar:.2f}_theta0_{theta0:.2f}_"
                       f"rep_{replicate:02d}")
                rows.append({
                    "task_id": len(rows),
                    "tier": tier,
                    "alpha": alpha,
                    "aspect_ratio": ar,
                    "theta0": theta0,
                    "target_theta": "" if target is None else target,
                    "replicate": replicate,
                    "seed": 260910 + 1000 * case_index + 10 * start_index + replicate,
                    "particles": args.particles,
                    "tau_end": args.tau_end,
                    "output_prefix": str(Path(args.results) / tag),
                })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} HCS tasks to {output}")


if __name__ == "__main__":
    main()
