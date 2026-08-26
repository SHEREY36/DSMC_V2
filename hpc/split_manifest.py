#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with open(args.manifest, newline="") as handle:
        reader = csv.DictReader(handle); rows, fields = list(reader), reader.fieldnames
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), args.max_rows):
        path = output / f"{Path(args.manifest).stem}_part_{start // args.max_rows:03d}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for local_id, row in enumerate(rows[start:start + args.max_rows]):
                row = dict(row); row["task_id"] = local_id; writer.writerow(row)
        print(path)


if __name__ == "__main__":
    main()

