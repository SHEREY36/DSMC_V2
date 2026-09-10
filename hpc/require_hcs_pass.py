#!/usr/bin/env python3
"""Release excitation only after the baseline HCS physics gate passes."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    args = parser.parse_args()
    payload = json.loads(Path(args.summary).read_text())
    if not payload.get("physics_gate_pass", False):
        raise SystemExit("baseline HCS physics gate has not passed; excitation is blocked")
    print("baseline HCS physics gate passed; excitation pilot is released")


if __name__ == "__main__":
    main()
