#!/usr/bin/env python3
"""Compile one independent artifact node for the Slurm precompute array."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from coll_models_v2.artifact import precompute_artifact_node
from coll_models_v2.legacy_bl import LegacyBL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--node-estimates", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--propensity-offsets", type=int, default=128)
    parser.add_argument("--gamma-max-table", required=True)
    parser.add_argument("--one-hit-table", required=True)
    parser.add_argument("--coefficient-rows")
    parser.add_argument("--coefficient-release-policy", default="strict")
    parser.add_argument("--beta-a", type=float, default=1.21)
    parser.add_argument("--beta-b", type=float, default=3.67)
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave a node whose payload is already written (resubmission)")
    args = parser.parse_args()
    done = Path(args.output) / f"node_{int(args.index):04d}.npz"
    if args.skip_existing and done.is_file():
        print(f"artifact node {args.index} already written: {done}", flush=True)
        return

    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "output_directory" not in rows[0]:
        parser.error("manifest must contain output_directory rows")
    runs = [row["output_directory"] for row in rows]
    bl = LegacyBL.load(args.gamma_max_table, args.one_hit_table,
                       args.beta_a, args.beta_b)
    target = precompute_artifact_node(
        runs, args.node_estimates, args.output, args.index, bl,
        propensity_offsets=args.propensity_offsets,
        propensity_workers=args.workers,
        coefficient_rows_path=args.coefficient_rows,
        coefficient_release_policy=args.coefficient_release_policy)
    print(f"artifact node {args.index} complete: {target}", flush=True)


if __name__ == "__main__":
    main()
