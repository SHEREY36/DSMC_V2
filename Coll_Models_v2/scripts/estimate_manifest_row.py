#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from coll_models_v2.estimate import estimate_node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[args.index]
    result = estimate_node(row["run_directories"].split(";"), args.bootstrap)
    path = Path(args.output) / (
        f"alpha_{float(row['alpha']):.3f}_theta_{float(row['theta']):.3f}_"
        f"AR_{float(row['aspect_ratio']):.3f}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

