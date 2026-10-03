#!/usr/bin/env python3
"""Fit one virtual excitation by exact incoming-pair importance sampling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from coll_models_v2.estimate import NODE_ESTIMATE_CONTRACT
from coll_models_v2.excitation import estimate_excitation


SCIENTIFIC_EXCEPTIONS = (ValueError, ArithmeticError, RuntimeError)


def digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--propensity-offsets", type=int, default=128)
    args = parser.parse_args()

    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= args.index < len(rows):
        raise IndexError(f"excitation task {args.index} outside 0..{len(rows)-1}")
    row = rows[args.index]
    baseline_path = Path(row["baseline_estimate"])
    baseline = json.loads(baseline_path.read_text())
    expected = tuple(float(row[name]) for name in ("alpha", "theta", "aspect_ratio"))
    actual = tuple(float(baseline[name]) for name in ("alpha", "theta", "aspect_ratio"))
    if actual != expected or int(baseline.get("ensemble_id", 0)) != 0:
        raise ValueError("excitation manifest does not match its baseline estimate")
    if baseline.get("estimator_contract") != NODE_ESTIMATE_CONTRACT \
            or not baseline.get("qa", {}).get("precision_pass", False):
        raise ValueError("excitation baseline is stale or has not passed closure QA")
    if {Path(path).name for path in baseline.get("source_runs", [])} \
            != {Path(row["baseline_directory"]).name}:
        raise ValueError("excitation baseline raw-shard identity changed")

    provenance = {
        "manifest": str(Path(args.manifest)), "manifest_index": int(args.index),
        "baseline_estimate": str(baseline_path),
        "baseline_estimate_digest": digest(baseline),
        "bootstrap": int(args.bootstrap),
        "propensity_offsets": int(args.propensity_offsets),
    }
    try:
        result = estimate_excitation(
            [row["baseline_directory"]], row["family"], float(row["eta"]),
            int(row["ensemble_id"]),
            (float(baseline["energy"].get("anchor_c1", 0.0)),
             float(baseline["energy"].get("anchor_c2", 0.0))),
            n_bootstrap=args.bootstrap,
            propensity_offsets=args.propensity_offsets,
            bootstrap_seed=20260910 + args.index,
            kernel_form=baseline["energy"]["kernel_form"],
        )
        result["excitation_provenance"] = provenance
        result["excitation_status"] = "pass"
    except SCIENTIFIC_EXCEPTIONS as exc:
        # Overlap or model-form failures are campaign results, not lost Slurm
        # work.  The summary remains fail-closed and will not release them.
        result = {
            "schema_version": "excitation-failure-v1",
            "alpha": expected[0], "theta": expected[1],
            "aspect_ratio": expected[2],
            "ensemble_id": int(row["ensemble_id"]),
            "excitation": {"family": row["family"], "eta": float(row["eta"])},
            "excitation_status": "blocked",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "excitation_provenance": provenance,
        }
    target = Path(row["output_file"])
    atomic_json(target, result)
    print(f"excitation row {args.index}: {target} ({result['excitation_status']})",
          flush=True)


if __name__ == "__main__":
    main()
