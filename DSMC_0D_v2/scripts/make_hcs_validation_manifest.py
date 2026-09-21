#!/usr/bin/env python3
"""Create the compact HCS gate that precedes excitation/USF validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FIELDS = (
    "task_id", "campaign_mode", "tier", "alpha", "aspect_ratio", "theta0", "target_theta",
    "replicate", "seed", "particles", "tau_end", "state_update_cpp",
    "output_prefix",
)


def full_domain_cases(artifact: Path) -> tuple[tuple[str, float, float, float | None], ...]:
    """Return every calibrated alpha/AR axis pair in the artifact.

    This is deliberately driven by the artifact itself instead of a manually
    maintained list, so an expanded gate cannot silently omit new nodes.
    """
    with np.load(artifact, allow_pickle=False) as data:
        coordinates = np.asarray(data["surface_coordinates"], dtype=float)
    alphas = np.unique(coordinates[:, 0])
    aspect_ratios = np.unique(coordinates[:, 2])
    known = {(alpha, ar): target for _, alpha, ar, target in GATE_CASES}
    cases = []
    for alpha in alphas:
        for ar in aspect_ratios:
            target = known.get((float(alpha), float(ar)))
            # Equipartition is an exact target in the elastic limit for every
            # nonspherical geometry, whether or not a DEM datum was tabulated.
            if np.isclose(alpha, 1.0):
                target = 1.0
            tier = "gate" if target is not None else "domain"
            cases.append((tier, float(alpha), float(ar), target))
    return tuple(cases)


def learned_extreme_starts(artifact: Path) -> dict[tuple[float, float], tuple[float, ...]]:
    """Return exact learned theta nodes far from equipartition for each case.

    The low-temperature-ratio surface is deliberately denser near the sphere.
    Read the actual sparse coordinates rather than manufacturing a Cartesian
    grid.  ``theta <= 0.2`` and ``theta >= 2`` are the declared far-from-one
    nodes used by this diagnostic.
    """
    with np.load(artifact, allow_pickle=False) as data:
        coordinates = np.asarray(data["surface_coordinates"], dtype=float)
    result: dict[tuple[float, float], tuple[float, ...]] = {}
    for alpha, ar in sorted({(float(row[0]), float(row[2]))
                             for row in coordinates}):
        starts = sorted({float(row[1]) for row in coordinates
                         if np.isclose(row[0], alpha)
                         and np.isclose(row[2], ar)
                         and (row[1] <= 0.2 + 1.0e-12
                              or row[1] >= 2.0 - 1.0e-12)})
        if len(starts) < 2:
            raise ValueError(
                f"learned-extremes case {(alpha, ar)} has fewer than two starts")
        result[(alpha, ar)] = tuple(starts)
    return result

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
    parser.add_argument(
        "--state-update-cpp", type=float, default=0.05,
        help=("collisions per particle between cell-feature refreshes; "
              "use 0 for the exact every-time-step reference"))
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--mode", choices=("compact", "evidence", "full-domain",
                                            "learned-extremes"),
                        default="compact")
    parser.add_argument("--artifact")
    parser.add_argument(
        "--case", action="append", default=[], metavar="ALPHA,AR",
        help="retain only one alpha,aspect-ratio pair; may be repeated")
    parser.add_argument(
        "--theta0", type=float, action="append", default=[],
        help=("retain selected learned initial theta values; valid only for "
              "--mode learned-extremes and may be repeated"))
    args = parser.parse_args()

    if args.replicates < 1:
        parser.error("--replicates must be positive")
    if args.state_update_cpp < 0.0:
        parser.error("--state-update-cpp must be nonnegative")
    rows = []
    learned_starts = None
    if args.mode in ("full-domain", "learned-extremes"):
        if not args.artifact:
            parser.error(f"--artifact is required for --mode {args.mode}")
        artifact = Path(args.artifact)
        if not artifact.is_file():
            parser.error(f"artifact does not exist: {artifact}")
        cases = full_domain_cases(artifact)
        if args.mode == "learned-extremes":
            learned_starts = learned_extreme_starts(artifact)
    else:
        if args.theta0:
            parser.error("--theta0 is valid only for --mode learned-extremes")
        cases = GATE_CASES + (DIAGNOSTIC_CASES if args.include_diagnostics else ())
    if args.case:
        requested = set()
        for value in args.case:
            try:
                alpha_text, ar_text = value.split(",", 1)
                requested.add((float(alpha_text), float(ar_text)))
            except ValueError as error:
                parser.error(f"invalid --case {value!r}; expected ALPHA,AR")
        cases = tuple(case for case in cases if (case[1], case[2]) in requested)
        found = {(case[1], case[2]) for case in cases}
        if found != requested:
            parser.error(f"requested HCS cases are unavailable: {sorted(requested - found)}")
    requested_starts = set(args.theta0)
    for case_index, (tier, alpha, ar, target) in enumerate(cases):
        starts = ((0.75, 1.25) if learned_starts is None
                  else learned_starts[(alpha, ar)])
        if requested_starts:
            starts = tuple(theta for theta in starts
                           if any(np.isclose(theta, requested)
                                  for requested in requested_starts))
        if not starts:
            continue
        for start_index, theta0 in enumerate(starts):
            for replicate in range(args.replicates):
                tag = (f"alpha_{alpha:.2f}_AR_{ar:.2f}_theta0_{theta0:.2f}_"
                       f"rep_{replicate:02d}")
                rows.append({
                    "task_id": len(rows),
                    "campaign_mode": args.mode,
                    "tier": tier,
                    "alpha": alpha,
                    "aspect_ratio": ar,
                    "theta0": theta0,
                    "target_theta": "" if target is None else target,
                    "replicate": replicate,
                    "seed": 260910 + 1000 * case_index + 10 * start_index + replicate,
                    "particles": args.particles,
                    "tau_end": args.tau_end,
                    "state_update_cpp": args.state_update_cpp,
                    "output_prefix": str(Path(args.results) / tag),
                })

    if not rows:
        parser.error("validation filters selected no HCS rows")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} HCS tasks to {output}")


if __name__ == "__main__":
    main()
