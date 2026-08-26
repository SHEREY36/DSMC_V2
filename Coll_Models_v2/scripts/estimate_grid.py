#!/usr/bin/env python3
import argparse

from coll_models_v2.pipeline import estimate_grid
from coll_models_v2.legacy_bl import LegacyBL


def main():
    parser = argparse.ArgumentParser(description="Estimate every finalized CTC v2 grid node")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--gamma-max-table", required=True)
    parser.add_argument("--one-hit-table", required=True)
    parser.add_argument("--beta-a", type=float, default=1.21)
    parser.add_argument("--beta-b", type=float, default=3.67)
    args = parser.parse_args()
    bl = LegacyBL.load(args.gamma_max_table, args.one_hit_table,
                       args.beta_a, args.beta_b)
    rows = estimate_grid(args.runs_root, args.output, bl, args.bootstrap)
    print(f"Estimated {len(rows)} node(s)")


if __name__ == "__main__":
    main()
