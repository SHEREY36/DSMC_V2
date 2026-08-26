#!/usr/bin/env python3
import argparse

from coll_models_v2.artifact import build_artifact
from coll_models_v2.pipeline import discover_runs
from coll_models_v2.legacy_bl import LegacyBL


def main():
    parser = argparse.ArgumentParser(description="Build conservative microscopic_closure_v2")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--runs-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--gamma-max-table", required=True)
    parser.add_argument("--one-hit-table", required=True)
    parser.add_argument("--beta-a", type=float, default=1.21)
    parser.add_argument("--beta-b", type=float, default=3.67)
    args = parser.parse_args()
    runs = args.runs or (discover_runs(args.runs_root) if args.runs_root else [])
    if not runs:
        parser.error("provide run directories or --runs-root")
    bl = LegacyBL.load(args.gamma_max_table, args.one_hit_table,
                       args.beta_a, args.beta_b)
    result = build_artifact(runs, args.output, bl, args.bootstrap)
    print(f"Wrote {result['artifact_type']} with {result['n_nodes']} node(s) to {args.output}")


if __name__ == "__main__":
    main()
