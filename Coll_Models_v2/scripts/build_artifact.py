#!/usr/bin/env python3
import argparse
import csv

from coll_models_v2.artifact import build_artifact
from coll_models_v2.pipeline import discover_runs
from coll_models_v2.legacy_bl import LegacyBL


def main():
    parser = argparse.ArgumentParser(description="Build conservative microscopic_closure_v2")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--runs-root")
    parser.add_argument("--manifest",
                        help="canonical CSV whose output_directory column is the exact run set")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--gamma-max-table", required=True)
    parser.add_argument("--one-hit-table", required=True)
    parser.add_argument("--node-estimates",
                        help="directory of QA-passed per-node estimator JSON files")
    parser.add_argument("--beta-a", type=float, default=1.21)
    parser.add_argument("--beta-b", type=float, default=3.67)
    args = parser.parse_args()
    if args.runs and (args.runs_root or args.manifest):
        parser.error("positional runs, --runs-root, and --manifest are mutually exclusive")
    if args.runs_root and args.manifest:
        parser.error("--runs-root and --manifest are mutually exclusive")
    if args.manifest:
        with open(args.manifest, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "output_directory" not in rows[0]:
            parser.error("--manifest must contain at least one output_directory row")
        runs = [row["output_directory"] for row in rows]
    else:
        runs = args.runs or (discover_runs(args.runs_root) if args.runs_root else [])
    if not runs:
        parser.error("provide run directories, --runs-root, or --manifest")
    bl = LegacyBL.load(args.gamma_max_table, args.one_hit_table,
                       args.beta_a, args.beta_b)
    result = build_artifact(runs, args.output, bl, args.bootstrap, args.node_estimates)
    print(f"Wrote {result['artifact_type']} with {result['n_nodes']} node(s) to {args.output}")


if __name__ == "__main__":
    main()
