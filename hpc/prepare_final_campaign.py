#!/usr/bin/env python3
"""Grid and fit manifests for the final closure: every node refitted.

The grid is the alpha-refined grid plus the new theta planes.  Every node is
fitted again because the exchange kernel is now fitted with the post-collision
energy weight (estimator contract v4); a node keeps its kernel form and anchor,
and a new theta node takes the anchor of its aspect ratio.  The fit manifest is
the whole grid, so a fit array index is the grid row.
"""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-grid", required=True)
    parser.add_argument("--theta-manifest", required=True)
    parser.add_argument("--grid", required=True)
    args = parser.parse_args()
    base = list(csv.DictReader(open(args.base_grid, newline="")))
    fields = list(base[0])
    anchor = {}
    for row in base:
        anchor.setdefault(round(float(row["aspect_ratio"]), 4), row["anchor_directory"])
    rows = [dict(row) for row in base]
    for row in csv.DictReader(open(args.theta_manifest, newline="")):
        row = {key: row.get(key, "") for key in fields}
        row["kernel_form"] = ""
        row["anchor_directory"] = anchor[round(float(row["aspect_ratio"]), 4)]
        rows.append(row)
    keys = [(round(float(r["alpha"]), 4), round(float(r["theta"]), 5),
             round(float(r["aspect_ratio"]), 4), int(r["ensemble_id"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate nodes in the merged grid")
    for i, row in enumerate(rows):
        row["task_id"] = i
    Path(args.grid).parent.mkdir(parents=True, exist_ok=True)
    with open(args.grid, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"{args.grid}: {len(rows)} nodes ({len(base)} existing + {len(rows) - len(base)} new theta)")


if __name__ == "__main__":
    main()
