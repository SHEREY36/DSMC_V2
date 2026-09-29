#!/usr/bin/env python3
"""Validate harvested reservoirs and create exact replay CTC tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from dsmc_v2_contracts import FEATURE_NAMES


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--replay-samples", type=int, default=40_000)
    args = parser.parse_args()
    with Path(args.harvest_manifest).open(newline="") as handle:
        harvest = list(csv.DictReader(handle))
    if len(harvest) != 42:
        raise SystemExit(f"expected the frozen 42-task harvest, found {len(harvest)}")
    root = Path(args.results)
    rows: list[dict[str, object]] = []
    verified_artifacts: dict[str, str] = {}
    used_metadata: set[Path] = set()
    used_binaries: set[Path] = set()
    for harvest_row in harvest:
        success = Path(harvest_row["output_prefix"] + "_SUCCESS.json")
        if not success.is_file():
            raise SystemExit(f"missing successful harvest: {success}")
        summary = json.loads(success.read_text())
        expected_task = int(harvest_row["task_id"])
        if (int(summary.get("task_id", -1)) != expected_task
                or summary.get("protocol_version")
                != harvest_row["protocol_version"]
                or summary.get("tag") != harvest_row["tag"]
                or summary.get("initial_branch")
                != harvest_row["initial_branch"]
                or not math.isclose(float(summary.get("alpha", -1.0)),
                                    float(harvest_row["alpha"]),
                                    rel_tol=0.0, abs_tol=1.0e-12)
                or not math.isclose(float(summary.get("aspect_ratio", -1.0)),
                                    float(harvest_row["aspect_ratio"]),
                                    rel_tol=0.0, abs_tol=1.0e-12)
                or summary.get("artifact") != harvest_row["artifact"]
                or summary.get("artifact_sha256")
                != harvest_row["artifact_sha256"]):
            raise SystemExit(f"harvest provenance mismatch: {success}")
        artifact = Path(harvest_row["artifact"])
        expected_hash = harvest_row["artifact_sha256"]
        if str(artifact) not in verified_artifacts:
            verified_artifacts[str(artifact)] = digest(artifact)
        actual_hash = verified_artifacts[str(artifact)]
        if actual_hash != expected_hash:
            raise SystemExit(f"frozen base artifact changed: {artifact}")
        if len(summary.get("reservoirs", [])) != 4:
            raise SystemExit(f"harvest does not contain four windows: {success}")
        expected_windows = [tuple(map(float, value.split(":")))
                            for value in harvest_row["windows"].split(";")]
        for window, item in enumerate(summary["reservoirs"]):
            metadata_path = Path(item["metadata"])
            binary_path = Path(item["binary"])
            if not metadata_path.is_file() or not binary_path.is_file():
                raise SystemExit(f"harvest reservoir file is missing: {metadata_path}")
            metadata_key = metadata_path.resolve()
            binary_key = binary_path.resolve()
            if metadata_key in used_metadata or binary_key in used_binaries:
                raise SystemExit(
                    f"duplicate replay path across harvest windows: {metadata_path}")
            used_metadata.add(metadata_key)
            used_binaries.add(binary_key)
            source = json.loads(metadata_path.read_text())
            provenance = source.get("source_provenance", {})
            expected_window = expected_windows[window]
            actual_window = tuple(map(float, source.get("tau_window", ())))
            if (source.get("schema_version") != "dsmc-ctc-replay-v1"
                    or source.get("record_dtype")
                    != "little_endian_float64"
                    or int(source.get("record_width", -1)) != 18
                    or source.get("sampling_contract")
                    != "post_ntc_pre_orientation_collision_flux_v1"
                    or source.get("cell_measure_debiasing")
                    != "inverse_physical_selection_speed_v1"
                    or int(source.get("window_index", -1)) != window
                    or provenance.get("protocol_version")
                    != harvest_row["protocol_version"]
                    or provenance.get("tag") != harvest_row["tag"]
                    or int(provenance.get("task_id", -1)) != expected_task
                    or provenance.get("initial_branch")
                    != harvest_row["initial_branch"]
                    or len(actual_window) != 2
                    or any(not math.isclose(actual, expected, rel_tol=0.0,
                                            abs_tol=1.0e-9)
                           for actual, expected in zip(
                               actual_window, expected_window))
                    or not math.isclose(float(source.get("alpha", -1.0)),
                                        float(harvest_row["alpha"]),
                                        rel_tol=0.0, abs_tol=1.0e-12)
                    or not math.isclose(
                        float(source.get("aspect_ratio", -1.0)),
                        float(harvest_row["aspect_ratio"]),
                        rel_tol=0.0, abs_tol=1.0e-12)
                    or Path(source.get("binary_file", "")).resolve()
                    != binary_key
                    or item.get("binary_sha256")
                    != source.get("binary_sha256")
                    or int(item.get("record_count", -1))
                    != int(source.get("record_count", -2))):
                raise SystemExit(
                    f"harvest reservoir provenance mismatch: {metadata_path}")
            if digest(binary_path) != source.get("binary_sha256"):
                raise SystemExit(
                    f"harvest replay binary hash mismatch: {binary_path}")
            expected_bytes = int(source["record_count"]) * 18 * 8
            if binary_path.stat().st_size != expected_bytes:
                raise SystemExit(
                    f"harvest replay binary size mismatch: {binary_path}")
            cell_features = source.get("cell_features")
            domain_features = source.get("domain_features")
            if (not isinstance(cell_features, dict)
                    or not isinstance(domain_features, dict)
                    or set(cell_features) != set(FEATURE_NAMES)
                    or set(domain_features) != set(FEATURE_NAMES)
                    or any(not math.isfinite(float(value))
                           for value in (*cell_features.values(),
                                         *domain_features.values()))
                    or not math.isfinite(float(source.get("theta", -1.0)))
                    or float(source.get("theta", -1.0)) <= 0.0):
                raise SystemExit(
                    f"harvest replay state features are invalid: {metadata_path}")
            if not source.get("qa", {}).get("pass", False):
                raise SystemExit(f"reservoir failed QA: {metadata_path}")
            if int(source["record_count"]) < args.replay_samples:
                raise SystemExit(f"reservoir is too small: {metadata_path}")
            ar = float(source["aspect_ratio"])
            branch = harvest_row["initial_branch"]
            output = root / "replay" / (
                f"AR_{ar:.3f}_alpha_{float(source['alpha']):.3f}_{branch}_"
                f"window_{window:02d}")
            rows.append({
                "task_id": len(rows), "role": "usf_replay",
                "alpha": f"{float(source['alpha']):.12g}",
                "theta": f"{float(source['theta']):.16g}",
                "aspect_ratio": f"{ar:.12g}",
                "seed": 280903000 + len(rows),
                "nsamples": args.replay_samples,
                "ensemble_id": len(rows),
                "source_metadata": str(metadata_path),
                "replay_file": str(source["binary_file"]),
                "initial_branch": branch, "window_index": window,
                "artifact": str(artifact), "artifact_sha256": expected_hash,
                "output_directory": str(output),
            })
    if len(rows) != 168:
        raise RuntimeError(f"expected 168 replay tasks, got {len(rows)}")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} CTC tasks to {target}")


if __name__ == "__main__":
    main()
