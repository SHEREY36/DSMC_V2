#!/usr/bin/env python3
"""Validate harvested reservoirs and create exact replay CTC tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


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
    for harvest_row in harvest:
        success = Path(harvest_row["output_prefix"] + "_SUCCESS.json")
        if not success.is_file():
            raise SystemExit(f"missing successful harvest: {success}")
        summary = json.loads(success.read_text())
        if (summary.get("artifact") != harvest_row["artifact"]
                or summary.get("artifact_sha256")
                != harvest_row["artifact_sha256"]):
            raise SystemExit(f"harvest/base-artifact provenance mismatch: {success}")
        artifact = Path(harvest_row["artifact"])
        expected_hash = harvest_row["artifact_sha256"]
        if str(artifact) not in verified_artifacts:
            verified_artifacts[str(artifact)] = digest(artifact)
        actual_hash = verified_artifacts[str(artifact)]
        if actual_hash != expected_hash:
            raise SystemExit(f"frozen base artifact changed: {artifact}")
        if len(summary.get("reservoirs", [])) != 4:
            raise SystemExit(f"harvest does not contain four windows: {success}")
        for window, item in enumerate(summary["reservoirs"]):
            metadata_path = Path(item["metadata"])
            source = json.loads(metadata_path.read_text())
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
