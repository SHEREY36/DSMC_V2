#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coll_models_v2.estimate import estimate_node


def main():
    parser = argparse.ArgumentParser(description="Estimate one v2 collision-operator node")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = estimate_node(args.runs, n_bootstrap=args.bootstrap)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

