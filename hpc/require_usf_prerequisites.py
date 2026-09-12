#!/usr/bin/env python3
"""Fail closed unless HCS and independent fresh-CTC validation passed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: str) -> dict:
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"missing prerequisite: {source}")
    return json.loads(source.read_text())


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hcs", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--pilot-summary")
    args = parser.parse_args()
    hcs = load(args.hcs)
    holdout = load(args.holdout)
    artifact = Path(args.artifact)
    if not artifact.is_file():
        raise SystemExit(f"missing candidate artifact: {artifact}")
    artifact_digest = sha256(artifact)
    if not hcs.get("physics_gate_pass", False):
        raise SystemExit("corrected HCS physics gate has not passed")
    if hcs.get("artifact_sha256") != artifact_digest:
        raise SystemExit("HCS gate was not run with these candidate artifact bytes")
    if not holdout.get("validation_pass", False):
        raise SystemExit("independent fresh-CTC response validation has not passed")
    if holdout.get("artifact_sha256") != artifact_digest:
        raise SystemExit("fresh-CTC holdout did not score these candidate artifact bytes")
    if args.pilot_summary:
        pilot = load(args.pilot_summary)
        if not pilot.get("full_validation_ready", False):
            raise SystemExit("USF pilot did not authorize the full validation grid")
        if pilot.get("artifact_sha256") != artifact_digest:
            raise SystemExit("USF pilot did not use these candidate artifact bytes")
    print("USF prerequisites pass: HCS, fresh CTC, and artifact are consistent")


if __name__ == "__main__":
    main()
