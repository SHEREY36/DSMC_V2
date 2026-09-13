#!/usr/bin/env python3
"""Build small, decision-making importance-sampled excitation campaigns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from coll_models_v2.excitation import EXCITATION_FAMILIES
from coll_models_v2.response import CENTRAL_AMPLITUDE


HCS_FAMILIES = ("a2_tr", "a2_rot", "a11", "A_cu")
AMPLITUDES = (-0.50, -0.25, 0.25, 0.50)
FIELDS = (
    "task_id", "mode", "alpha", "theta", "aspect_ratio", "ensemble_id",
    "family", "target_feature", "eta", "baseline_directory",
    "baseline_estimate", "output_file",
)


def coordinate(row) -> tuple[float, float, float]:
    return tuple(float(row[name]) for name in ("alpha", "theta", "aspect_ratio"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hcs-pilot", "full-pilot",
                                            "correction-grid", "production-grid",
                                            "independent-holdout", "usf-extension",
                                            "near-sphere-extension",
                                            "support-refinement"),
                        default="hcs-pilot")
    parser.add_argument("--grid", default="manifests/artifact_grid.csv")
    parser.add_argument("--estimates", default="results/closure_estimates/artifact_grid")
    parser.add_argument("--output", default="manifests/excitation_pilot.csv")
    parser.add_argument("--results", default="results/closure_estimates/excitation_pilot")
    parser.add_argument(
        "--exclude-manifest", action="append", default=[],
        help="for production-grid, omit physical nodes already present here")
    parser.add_argument(
        "--include-manifest", action="append", default=[],
        help="for support-refinement, use the physical nodes present here")
    parser.add_argument("--amplitudes", nargs="+", type=float)
    args = parser.parse_args()

    if args.mode == "hcs-pilot":
        requested = {(alpha, 1.0, ar)
                     for alpha in (0.80, 0.95, 1.00) for ar in (2.0, 3.0)}
        families = HCS_FAMILIES
    elif args.mode == "full-pilot":
        # One representative USF-domain node first.  This is deliberately a
        # rank/overlap/model-response pilot, not a premature 3-D campaign.
        requested = {(0.95, 1.0, 2.0)}
        families = EXCITATION_FAMILIES
    elif args.mode == "correction-grid":
        # Small 3-D tensor grid spanning the six truth-backed HCS points and
        # two off-equilibrium theta planes. It is the candidate-artifact gate,
        # not yet the full production surface.
        requested = {(alpha, theta, ar)
                     for alpha in (0.80, 0.95, 1.00)
                     for theta in (0.20, 1.00, 2.00)
                     for ar in (2.0, 3.0)}
        families = EXCITATION_FAMILIES
    elif args.mode == "usf-extension":
        # Minimal union of exact artifact-grid planes needed to enlarge the
        # correction hull from alpha >= 0.8, AR >= 2 to the held-out USF
        # domain alpha >= 0.5, AR >= 1.5.  AR=2.5 and intermediate alpha
        # values are then interpolation points, not additional fit nodes.
        requested = (
            {(0.50, theta, ar)
             for theta in (0.20, 1.00, 2.00) for ar in (2.00, 3.00)}
            | {(alpha, theta, 1.50)
               for alpha in (0.50, 0.80, 0.95, 1.00)
               for theta in (0.20, 1.00, 2.00)}
        )
        families = EXCITATION_FAMILIES
    elif args.mode == "near-sphere-extension":
        # Complete the correction hull down to the smallest nonspherical
        # baseline nodes.  AR=1 is the separate exact sphere kernel; the
        # variational spherocylinder artifact starts at AR=1.1.
        requested = {(alpha, theta, ar)
                     for alpha in (0.50, 0.80, 0.95, 1.00)
                     for theta in (0.20, 1.00, 2.00)
                     for ar in (1.10, 1.20, 1.35)}
        families = EXCITATION_FAMILIES
    elif args.mode == "support-refinement":
        requested = set()
        for source in args.include_manifest:
            with open(source, newline="") as handle:
                requested.update(coordinate(row) for row in csv.DictReader(handle))
        if not requested:
            raise ValueError("support-refinement requires --include-manifest")
        families = EXCITATION_FAMILIES
    elif args.mode == "independent-holdout":
        # A fresh CTC shard at this representative inelastic node is not used
        # in fitting the artifact.  The two boundary amplitudes are therefore
        # validation observations only; regenerating the central training
        # amplitudes would add cost without making the holdout more independent.
        requested = {(0.95, 1.0, 2.0)}
        families = EXCITATION_FAMILIES
    else:
        # Final surface: only justified after the small correction grid passes
        # corrected HCS and direct-CTC validation.
        requested = None
        families = EXCITATION_FAMILIES

    with open(args.grid, newline="") as handle:
        grid = list(csv.DictReader(handle))
    if requested is None:
        requested = {coordinate(row) for row in grid}
    excluded = set()
    for source in args.exclude_manifest:
        with open(source, newline="") as handle:
            excluded.update(coordinate(row) for row in csv.DictReader(handle))
    requested -= excluded
    if not requested:
        raise ValueError("no physical nodes remain after exclusions")
    selected = {coordinate(row): row for row in grid if coordinate(row) in requested}
    missing = sorted(requested - set(selected))
    if missing:
        raise ValueError(f"artifact grid has no exact baseline nodes for {missing}")

    estimates = {}
    for path in Path(args.estimates).glob("alpha_*.json"):
        payload = json.loads(path.read_text())
        if int(payload.get("ensemble_id", 0)) == 0:
            estimates[coordinate(payload)] = (path, payload)

    amplitudes = (tuple(args.amplitudes) if args.amplitudes is not None else
                  (-0.50, 0.50) if args.mode == "independent-holdout" else
                  (-0.35, 0.35) if args.mode == "support-refinement" else
                  AMPLITUDES)
    if args.mode == "support-refinement" and (
            not amplitudes or any(abs(value) <= CENTRAL_AMPLITUDE
                                  for value in amplitudes)):
        raise ValueError("support refinement amplitudes must exceed 0.25")
    rows = []
    results = Path(args.results)
    for physical in sorted(requested):
        if physical not in estimates:
            raise ValueError(f"missing baseline estimate at {physical}")
        estimate_path, estimate = estimates[physical]
        if not estimate.get("qa", {}).get("precision_pass", False):
            raise ValueError(f"baseline estimate at {physical} has not passed QA")
        for family_index, family in enumerate(families):
            for amplitude_index, eta in enumerate(amplitudes):
                ensemble_id = (1001 if args.mode == "support-refinement" else 1) \
                    + family_index * len(amplitudes) + amplitude_index
                tag = (f"alpha_{physical[0]:.3f}_theta_{physical[1]:.3f}_"
                       f"AR_{physical[2]:.3f}_ensemble_{ensemble_id:03d}.json")
                rows.append({
                    "task_id": len(rows), "mode": args.mode,
                    "alpha": physical[0], "theta": physical[1],
                    "aspect_ratio": physical[2], "ensemble_id": ensemble_id,
                    "family": family,
                    "target_feature": family.removesuffix("_opposed"),
                    "eta": eta,
                    "baseline_directory": selected[physical]["output_directory"],
                    "baseline_estimate": str(estimate_path),
                    "output_file": str(results / tag),
                })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} {args.mode} excitation tasks to {output}")


if __name__ == "__main__":
    main()
