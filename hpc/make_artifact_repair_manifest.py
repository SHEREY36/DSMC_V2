#!/usr/bin/env python3
"""Select failed canonical nodes for the logit-cubic repair fit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "python"))
sys.path.insert(0, str(ROOT / "Coll_Models_v2" / "src"))
from coll_models_v2.fit_exchange import LOGIT_CUBIC_KERNEL


def _key(values: dict) -> tuple[float, float, float, int]:
    return (float(values["alpha"]), float(values["theta"]),
            float(values["aspect_ratio"]), int(values.get("ensemble_id", 0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="manifests/artifact_grid.csv")
    parser.add_argument("--validation",
                        default="results/closure_estimates/artifact_grid_validation.json")
    parser.add_argument("--output", default="manifests/artifact_grid_repairs.csv")
    args = parser.parse_args()

    with open(args.grid, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        grid = {_key(row): row for row in reader}
    validation = json.loads(Path(args.validation).read_text())
    failures = validation.get("qa_failures", [])
    if not failures:
        raise ValueError("validation report contains no failed nodes to repair")

    rows = []
    for task_id, failure in enumerate(failures):
        key = tuple(failure["node"])
        if key not in grid:
            raise ValueError(f"failed node {key} is absent from the canonical grid")
        reasons = set(failure.get("reasons", []))
        if "model_form" not in reasons:
            raise ValueError(f"node {key} failed for {sorted(reasons)}, not model form")
        row = dict(grid[key])
        row["task_id"] = str(task_id)
        row["kernel_form"] = LOGIT_CUBIC_KERNEL
        rows.append(row)

    if "kernel_form" not in fields:
        fields.append("kernel_form")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"{output}: {len(rows)} targeted {LOGIT_CUBIC_KERNEL} refits")


if __name__ == "__main__":
    main()
