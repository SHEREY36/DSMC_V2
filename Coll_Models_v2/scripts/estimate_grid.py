#!/usr/bin/env python3
import argparse

from coll_models_v2.pipeline import estimate_grid


def main():
    parser = argparse.ArgumentParser(description="Estimate every finalized CTC v2 grid node")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    rows = estimate_grid(args.runs_root, args.output, args.bootstrap)
    print(f"Estimated {len(rows)} node(s)")


if __name__ == "__main__":
    main()

