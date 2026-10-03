#!/usr/bin/env python3
"""Validate and attach a DSMC replay source to a raw CTC run.

Run this after ``SphCyl`` and before ``finalize_run.py``.  It is intentionally
fail-closed: a CTC directory cannot be treated as direct USF evidence unless
the source bytes, thermodynamic coordinates, record count, and reservoir QA
all match the run that consumed them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    parser.add_argument("run_directory")
    parser.add_argument("source_metadata")
    args = parser.parse_args()

    run_directory = Path(args.run_directory)
    source_path = Path(args.source_metadata).resolve()
    target_path = run_directory / "metadata_v2.json"
    target = json.loads(target_path.read_text())
    source = json.loads(source_path.read_text())
    if source.get("schema_version") != "dsmc-ctc-replay-v1":
        raise SystemExit("unsupported replay metadata schema")
    if source.get("sampling_contract") \
            != "post_ntc_pre_orientation_collision_flux_v1":
        raise SystemExit("replay source has the wrong sampling contract")
    if not source.get("qa", {}).get("pass", False):
        raise SystemExit("replay source failed reservoir QA")
    binary = Path(source["binary_file"])
    if not binary.is_absolute():
        candidate = (source_path.parent / binary).resolve()
        binary = candidate if candidate.is_file() else binary.resolve()
    if not binary.is_file() or digest(binary) != source.get("binary_sha256"):
        raise SystemExit("replay binary is missing or its SHA-256 changed")

    checks = {
        "alpha": (float(target["alpha"]), float(source["alpha"])),
        "theta": (float(target["theta"]), float(source["theta"])),
        "aspect_ratio": (float(target["aspect_ratio"]),
                         float(source["aspect_ratio"])),
        "temperature_rotational": (
            float(target["temperature_rotational"]), 1.0),
    }
    mismatches = [name for name, values in checks.items()
                  if not np.isclose(*values, rtol=2.0e-10, atol=2.0e-10)]
    if mismatches:
        raise SystemExit("CTC/replay coordinate mismatch: " + ", ".join(mismatches))
    if target.get("sampling_mode") != "dsmc_post_ntc_replay_v1":
        raise SystemExit("CTC run does not declare replay sampling")
    if int(target.get("replay_record_count", -1)) != int(source["record_count"]):
        raise SystemExit("CTC/replay record counts differ")
    if int(target["nsamples"]) > int(source["record_count"]):
        raise SystemExit("CTC hit target exceeds unique replay records")

    target.update({
        "replay_source_metadata": str(source_path),
        "replay_source_metadata_sha256": digest(source_path),
        "replay_source_binary": str(binary),
        "replay_source_binary_sha256": source["binary_sha256"],
        "replay_source_sampling_contract": source["sampling_contract"],
        "replay_source_tau_window": source["tau_window"],
        "replay_source_cell_features": source["cell_features"],
        "replay_source_domain_features": source["domain_features"],
        "replay_source_qa": source["qa"],
    })
    atomic_json(target_path, target)
    print(f"attached verified replay provenance: {source_path}")


if __name__ == "__main__":
    main()
