#!/usr/bin/env python3
"""Merge campaign manifests into one canonical physical grid.

Input order is precedence order.  Exact duplicate physical nodes are retained
from the first manifest only.  Every row receives the same alpha=1, theta=1
anchor directory as every other row at its aspect ratio.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _key(row: dict) -> tuple[float, float, float, int]:
    return (float(row["alpha"]), float(row["theta"]),
            float(row["aspect_ratio"]), int(row.get("ensemble_id", 0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    selected: dict[tuple[float, float, float, int], dict] = {}
    fields: list[str] = []
    duplicate_count = 0
    for manifest in args.manifests:
        with open(manifest, newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fields:
                    fields.append(field)
            for source in reader:
                row = dict(source)
                key = _key(row)
                if key in selected:
                    duplicate_count += 1
                    continue
                if args.require_complete:
                    run = Path(row["output_directory"])
                    required = ("_SUCCESS", "metadata_v2.json", "attempts_v2.bin",
                                "outcomes_v2.bin", "qa_v2.json")
                    missing = [name for name in required if not (run / name).is_file()]
                    if missing:
                        raise FileNotFoundError(f"{run}: missing {', '.join(missing)}")
                selected[key] = row

    anchors: dict[float, str] = {}
    for key, row in selected.items():
        alpha, theta, ar, ensemble = key
        if ensemble == 0 and abs(alpha - 1.0) < 1.0e-12 \
                and abs(theta - 1.0) < 1.0e-12:
            anchors[ar] = row["output_directory"]
    aspect_ratios = {key[2] for key in selected}
    missing_anchors = sorted(aspect_ratios - set(anchors))
    if missing_anchors:
        raise ValueError(f"no canonical alpha=1, theta=1 anchor for AR {missing_anchors}")

    fields = [field for field in fields if field != "anchor_directory"]
    fields.append("anchor_directory")
    rows = []
    for task_id, (key, row) in enumerate(sorted(selected.items())):
        row["task_id"] = str(task_id)
        row["anchor_directory"] = anchors[key[2]]
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output}: {len(rows)} unique physical nodes; "
          f"removed {duplicate_count} duplicate rows; {len(anchors)} shared anchors")


if __name__ == "__main__":
    main()
