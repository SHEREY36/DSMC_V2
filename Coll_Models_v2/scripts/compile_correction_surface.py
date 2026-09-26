#!/usr/bin/env python3
"""Compile one provenance-checked coefficient surface for artifact arrays."""

from __future__ import annotations

import argparse
import json

from coll_models_v2.artifact import write_coefficient_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-estimates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--release-policy",
        choices=("strict", "validated-angular-only-v1"), default="strict")
    args = parser.parse_args()
    payload = write_coefficient_rows(
        args.node_estimates, args.output, release_policy=args.release_policy)
    print(json.dumps({key: payload[key] for key in (
        "schema", "release_policy", "n_source_nodes", "n_coefficient_nodes")},
        indent=2))


if __name__ == "__main__":
    main()
