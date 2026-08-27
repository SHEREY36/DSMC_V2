#!/usr/bin/env python3
"""Fail unless a complete combined QA summary is ready for artifact export."""

import argparse
import csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("qa_summary")
    parser.add_argument("--expected", type=int, default=870)
    args = parser.parse_args()
    with open(args.qa_summary, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != args.expected:
        raise ValueError(f"QA contains {len(rows)} nodes, expected {args.expected}")
    failed = [row for row in rows if row["precision_pass"].lower() != "true"]
    if failed:
        preview = [(row["alpha"], row["theta"], row["aspect_ratio"],
                    row["continuation_reasons"]) for row in failed[:10]]
        raise ValueError(f"{len(failed)} nodes still require continuation; first cases: {preview}")
    print(f"QA release gate passed for all {len(rows)} nodes")


if __name__ == "__main__":
    main()
