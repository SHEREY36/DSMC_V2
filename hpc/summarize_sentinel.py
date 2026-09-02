#!/usr/bin/env python3
"""Summarize measured Negishi throughput and sentinel acceptance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--estimates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runtime = [json.loads(path.read_text())
               for path in Path(args.runs_root).rglob("runtime_v2.json")]
    estimates = [json.loads(path.read_text())
                 for path in Path(args.estimates).glob("alpha_*.json")]
    if not runtime or not estimates:
        raise ValueError("sentinel summary requires runtime and node-estimate files")
    failures = [{"alpha": row["alpha"], "theta": row["theta"],
                 "aspect_ratio": row["aspect_ratio"],
                 "reasons": row["qa"].get("continuation_reasons", [])}
                for row in estimates if not row["qa"].get("sentinel_pass", False)]
    payload = {
        "n_runtime_shards": len(runtime), "n_estimated_nodes": len(estimates),
        "median_hits_per_second": float(np.median([row["hits_per_second"] for row in runtime])),
        "minimum_hits_per_second": float(np.min([row["hits_per_second"] for row in runtime])),
        "maximum_attempts_per_hit": float(np.max([row["attempts_per_hit"] for row in runtime])),
        "all_sentinel_gates_pass": not failures, "failures": failures,
        "release_full_baseline": not failures,
        "release_excitation_campaign": False,
        "excitation_release_reason": (
            "requires direct-ensemble sampler certification after baseline sentinel acceptance"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
