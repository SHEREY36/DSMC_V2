#!/usr/bin/env python3
"""Stop a dependent USF stage when the preceding scientific gate failed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--expected-mode", choices=("numerics", "pilot", "full"),
                        required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    if summary.get("protocol") != "usf-crossflow-v1":
        raise SystemExit("summary does not use the current USF protocol")
    if summary.get("mode") != args.expected_mode:
        raise SystemExit("summary mode does not match the requested stage")
    if not summary.get("complete") or summary.get("n_failures") != 0:
        raise SystemExit("USF stage is incomplete")
    if not summary.get("stage_pass"):
        raise SystemExit("USF scientific stage gate did not pass")
    print(f"{args.expected_mode} stage passed")


if __name__ == "__main__":
    main()
