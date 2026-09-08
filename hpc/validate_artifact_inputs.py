#!/usr/bin/env python3
"""Validate the canonical raw grid and its precomputed node estimates.

The default check is metadata/QA based and cheap enough for a login node.
``--deep`` replays the binary contract over every record and belongs in a
scheduled job for a large campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from coll_models_v2.estimate import NODE_ESTIMATE_CONTRACT
from dsmc_v2_contracts import load_run, validate_run


def _key(values: dict) -> tuple[float, float, float, int]:
    return (float(values["alpha"]), float(values["theta"]),
            float(values["aspect_ratio"]), int(values.get("ensemble_id", 0)))


def _estimate_name(key: tuple[float, float, float, int]) -> str:
    alpha, theta, ar, ensemble = key
    return (f"alpha_{alpha:.3f}_theta_{theta:.3f}_AR_{ar:.3f}_"
            f"ensemble_{ensemble:03d}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--estimates", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--require-current-estimates", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {_key(row): row for row in rows}
    duplicate_manifest_keys = len(rows) - len(expected)

    raw_failures = []
    raw_warnings = []
    schema_counts: Counter[str] = Counter()
    for key, row in expected.items():
        run_path = Path(row["output_directory"])
        required = ("_SUCCESS", "metadata_v2.json", "attempts_v2.bin",
                    "outcomes_v2.bin", "qa_v2.json")
        missing = [name for name in required if not (run_path / name).is_file()]
        if missing:
            raw_failures.append({"node": key, "reason": "missing_files", "files": missing})
            continue
        try:
            run = load_run(run_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raw_failures.append({"node": key, "reason": f"load_error:{exc}"})
            continue
        if _key(run.metadata) != key:
            raw_failures.append({"node": key, "reason": "metadata_identity_mismatch",
                                 "metadata_node": _key(run.metadata)})
        schema_counts[str(run.metadata.get("source_schema_version"))] += 1
        stored_qa = json.loads((run_path / "qa_v2.json").read_text())
        qa = validate_run(run) if args.deep else stored_qa
        if qa.get("status") != "pass":
            raw_failures.append({"node": key, "reason": "binary_contract_failure",
                                 "errors": qa.get("errors", [])})
        if qa.get("warnings"):
            raw_warnings.append({"node": key, "warnings": qa["warnings"]})

    available: dict[tuple[float, float, float, int], tuple[Path, dict]] = {}
    duplicate_estimate_keys = []
    for directory in args.estimates:
        for path in sorted(Path(directory).glob("alpha_*.json")):
            node = json.loads(path.read_text())
            key = _key(node)
            if key in available:
                duplicate_estimate_keys.append(key)
                continue
            available[key] = (path, node)

    estimate_failures = []
    qa_failures = []
    current_count = 0
    for key, row in expected.items():
        if key not in available:
            estimate_failures.append({"node": key, "reason": "missing_estimate",
                                      "expected_name": _estimate_name(key)})
            continue
        path, node = available[key]
        reasons = []
        if node.get("estimator_contract") != NODE_ESTIMATE_CONTRACT:
            reasons.append("stale_estimator_contract")
        missing_fields = [name for name in (
            "cell_features", "incoming_law", "incoming_law_energy") if name not in node]
        reasons.extend(f"missing_{name}" for name in missing_fields)
        expected_source = Path(row["output_directory"]).name
        actual_sources = {Path(value).name for value in node.get("source_runs", [])}
        if actual_sources != {expected_source}:
            reasons.append("source_shard_mismatch")
        if reasons:
            estimate_failures.append({"node": key, "file": str(path), "reasons": reasons})
        else:
            current_count += 1
        qa = node.get("qa", {})
        if not qa.get("precision_pass", qa.get("sentinel_pass", False)):
            qa_failures.append({"node": key, "file": str(path),
                                "reasons": qa.get("continuation_reasons") or
                                ["closure_qa_failure"]})

    estimates_required = bool(args.estimates) or args.require_current_estimates
    raw_pass = not raw_failures and duplicate_manifest_keys == 0
    estimates_current = (not estimates_required or
                         (not estimate_failures and current_count == len(expected)))
    closure_qa_pass = not estimates_required or not qa_failures
    passed = raw_pass and estimates_current and closure_qa_pass
    payload = {
        "manifest": str(Path(args.manifest)),
        "n_manifest_rows": len(rows),
        "n_unique_physical_nodes": len(expected),
        "duplicate_manifest_keys": duplicate_manifest_keys,
        "n_raw_valid": len(expected) - len(raw_failures),
        "raw_schema_versions": dict(schema_counts),
        "deep_binary_validation": bool(args.deep),
        "raw_pass": raw_pass,
        "raw_failures": raw_failures,
        "raw_warnings": raw_warnings,
        "n_estimates_found": len(available),
        "n_current_estimates": current_count,
        "duplicate_estimate_keys": duplicate_estimate_keys,
        "estimates_current": estimates_current,
        "estimate_failures": estimate_failures,
        "closure_qa_pass": closure_qa_pass,
        "qa_failures": qa_failures,
        "artifact_inputs_pass": passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact_inputs_pass": payload["artifact_inputs_pass"],
        "closure_qa_pass": payload["closure_qa_pass"],
        "deep_binary_validation": payload["deep_binary_validation"],
        "estimates_current": payload["estimates_current"],
        "n_current_estimates": payload["n_current_estimates"],
        "n_estimate_failures": len(payload["estimate_failures"]),
        "n_manifest_rows": payload["n_manifest_rows"],
        "n_qa_failures": len(payload["qa_failures"]),
        "n_raw_failures": len(payload["raw_failures"]),
        "n_raw_valid": payload["n_raw_valid"],
        "raw_pass": payload["raw_pass"],
    }, indent=2, sort_keys=True))

    if args.require_current_estimates and not estimates_current:
        raise SystemExit(2)
    if args.require_pass and not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
