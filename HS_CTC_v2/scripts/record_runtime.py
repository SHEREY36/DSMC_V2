#!/usr/bin/env python3
"""Attach wall-clock throughput diagnostics to one finalized CTC shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    if args.seconds <= 0.0:
        raise ValueError("wall time must be positive")
    directory = Path(args.directory)
    metadata = json.loads((directory / "metadata_v2.json").read_text())
    attempts, outcomes = int(metadata["n_attempts"]), int(metadata["n_outcomes"])
    payload = {
        "wall_seconds": args.seconds,
        "hits_per_second": outcomes / args.seconds,
        "attempts_per_second": attempts / args.seconds,
        "attempts_per_hit": attempts / outcomes,
        "n_attempts": attempts,
        "n_outcomes": outcomes,
    }
    (directory / "runtime_v2.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
