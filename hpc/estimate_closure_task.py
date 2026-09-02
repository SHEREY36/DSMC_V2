#!/usr/bin/env python3
"""Estimate one row of a schema-2.2 CTC campaign manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from coll_models_v2.estimate import estimate_node
from coll_models_v2.pipeline import precision_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[args.index]
    run = Path(row["output_directory"])
    if not (run / "_SUCCESS").is_file():
        raise FileNotFoundError(f"closure shard is not finalized: {run}")
    result = estimate_node([run], n_bootstrap=args.bootstrap,
                           bootstrap_seed=20260902 + args.index)
    passed, reasons = precision_status(result)
    result["qa"].update(precision_pass=passed, continuation_reasons=reasons)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / (
        f"alpha_{float(row['alpha']):.3f}_theta_{float(row['theta']):.3f}_"
        f"AR_{float(row['aspect_ratio']):.3f}_ensemble_{int(row['ensemble_id']):03d}.json")
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"task {args.index}: {target} ({'pass' if passed else 'continue'})")


if __name__ == "__main__":
    main()
