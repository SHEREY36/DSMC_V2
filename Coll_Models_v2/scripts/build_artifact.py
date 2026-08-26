#!/usr/bin/env python3
import argparse

from coll_models_v2.artifact import build_artifact
from coll_models_v2.pipeline import discover_runs


def main():
    parser = argparse.ArgumentParser(description="Build collision_operator_v2 from finalized CTC runs")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--runs-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    runs = args.runs or (discover_runs(args.runs_root) if args.runs_root else [])
    if not runs:
        parser.error("provide run directories or --runs-root")
    result = build_artifact(runs, args.output, args.bootstrap)
    print(f"Wrote {result['artifact_type']} with {result['n_nodes']} node(s) to {args.output}")


if __name__ == "__main__":
    main()
