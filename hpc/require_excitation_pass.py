#!/usr/bin/env python3
"""Fail closed unless the preceding excitation stage supports continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("--expected-mode")
    args = parser.parse_args()
    path = Path(args.summary)
    if not path.is_file():
        raise SystemExit(f"missing prerequisite excitation summary: {path}")
    payload = json.loads(path.read_text())
    modes = {node.get("mode") for node in payload.get("nodes", [])}
    if args.expected_mode and modes != {args.expected_mode}:
        raise SystemExit(f"expected {args.expected_mode} summary, found modes {sorted(modes)}")
    if not payload.get("screening_pass", False):
        raise SystemExit("prerequisite excitation screening did not pass")
    if not payload.get("response_fit_ready", False):
        raise SystemExit("prerequisite response fit/held-out linearity did not pass")
    print(f"excitation prerequisite passed: {path}")


if __name__ == "__main__":
    main()
