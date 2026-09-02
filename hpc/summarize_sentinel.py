#!/usr/bin/env python3
"""Summarize measured Negishi throughput and sentinel acceptance gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--estimates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.manifest, newline="") as handle:
        expected = list(csv.DictReader(handle))
    if not expected:
        raise ValueError("sentinel manifest is empty")

    runtime = []
    estimates = []
    failures = []
    for row in expected:
        alpha = float(row["alpha"])
        theta = float(row["theta"])
        aspect_ratio = float(row["aspect_ratio"])
        ensemble_id = int(row["ensemble_id"])
        identity = {
            "task_id": int(row["task_id"]),
            "alpha": alpha,
            "theta": theta,
            "aspect_ratio": aspect_ratio,
            "ensemble_id": ensemble_id,
        }
        runtime_path = Path(row["output_directory"]) / "runtime_v2.json"
        estimate_path = Path(args.estimates) / (
            f"alpha_{alpha:.3f}_theta_{theta:.3f}_AR_{aspect_ratio:.3f}_"
            f"ensemble_{ensemble_id:03d}.json"
        )
        reasons = []
        if runtime_path.is_file():
            runtime.append(json.loads(runtime_path.read_text()))
        else:
            reasons.append("missing_runtime_record")
        if estimate_path.is_file():
            estimate = json.loads(estimate_path.read_text())
            estimates.append(estimate)
            if not estimate.get("qa", {}).get("sentinel_pass", False):
                node_reasons = estimate.get("qa", {}).get("continuation_reasons")
                reasons.extend(node_reasons or ["sentinel_gate_failure"])
        else:
            reasons.append("missing_node_estimate")
        if reasons:
            failures.append({**identity, "reasons": sorted(set(reasons))})

    hits_per_second = [row["hits_per_second"] for row in runtime]
    attempts_per_hit = [row["attempts_per_hit"] for row in runtime]
    analysis_complete = len(estimates) == len(expected)
    runtime_complete = len(runtime) == len(expected)
    passed = analysis_complete and runtime_complete and not failures
    payload = {
        "n_expected_nodes": len(expected),
        "n_runtime_shards": len(runtime),
        "n_estimated_nodes": len(estimates),
        "analysis_complete": analysis_complete,
        "runtime_complete": runtime_complete,
        "median_hits_per_second": (
            None if not hits_per_second else float(np.median(hits_per_second))),
        "minimum_hits_per_second": (
            None if not hits_per_second else float(np.min(hits_per_second))),
        "maximum_attempts_per_hit": (
            None if not attempts_per_hit else float(np.max(attempts_per_hit))),
        "all_sentinel_gates_pass": passed,
        "failures": failures,
        "release_full_baseline": passed,
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
