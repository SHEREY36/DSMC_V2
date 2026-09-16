#!/usr/bin/env python3
"""Combine disjoint excitation manifests with fresh contiguous task IDs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    rows = []
    fieldnames = None
    seen = set()
    for source in args.inputs:
        with Path(source).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise SystemExit(f"manifest columns differ in {source}")
            for row in reader:
                identity = (row["alpha"], row["theta"], row["aspect_ratio"],
                            row["ensemble_id"])
                if identity in seen:
                    raise SystemExit(f"duplicate excitation observation {identity}")
                seen.add(identity)
                row["task_id"] = str(len(rows))
                rows.append(row)
    if not rows or fieldnames is None:
        raise SystemExit("no excitation rows supplied")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} unique excitation observations to {output}")


if __name__ == "__main__":
    main()
