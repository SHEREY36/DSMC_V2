#!/usr/bin/env python3
import argparse

from coll_models_v2.pipeline import estimate_grid


def main():
    parser = argparse.ArgumentParser(description="Estimate every finalized schema-2.2 CTC node")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    rows = estimate_grid(args.runs_root, args.output, n_bootstrap=args.bootstrap)
    print(f"Estimated {len(rows)} node(s)")


if __name__ == "__main__":
    main()
