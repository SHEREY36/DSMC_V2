#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coll_models_v2.estimate import estimate_node
from coll_models_v2.legacy_bl import LegacyBL


def main():
    parser = argparse.ArgumentParser(description="Estimate one v2 collision-operator node")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--gamma-max-table", required=True)
    parser.add_argument("--one-hit-table", required=True)
    parser.add_argument("--beta-a", type=float, default=1.21)
    parser.add_argument("--beta-b", type=float, default=3.67)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bl = LegacyBL.load(args.gamma_max_table, args.one_hit_table,
                       args.beta_a, args.beta_b)
    result = estimate_node(args.runs, bl, n_bootstrap=args.bootstrap)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
