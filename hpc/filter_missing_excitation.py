#!/usr/bin/env python3
"""Write a retry manifest containing only excitation outputs not yet present."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.manifest).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "output_file" not in fieldnames:
        raise SystemExit("excitation manifest has no output_file column")

    missing = []
    for row in rows:
        if not Path(row["output_file"]).is_file():
            row = dict(row)
            row["task_id"] = str(len(missing))
            missing.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(missing)
    print(f"wrote {len(missing)} missing excitation tasks to {output}")


if __name__ == "__main__":
    main()
