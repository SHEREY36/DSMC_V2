#!/usr/bin/env python3
"""Fail closed unless the USF study uses the frozen HCS-v8 evidence model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROTOCOL = "usf-crossflow-v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--hcs-summary", required=True)
    parser.add_argument("--reference-provenance", required=True)
    parser.add_argument("--numerics-manifest", required=True)
    parser.add_argument("--pilot-manifest", required=True)
    parser.add_argument("--full-manifest", required=True)
    args = parser.parse_args()

    artifact = Path(args.artifact)
    artifact_manifest_path = artifact.with_name("manifest.json")
    if not artifact.is_file() or not artifact_manifest_path.is_file():
        raise SystemExit("frozen artifact and adjacent manifest.json are required")
    artifact_hash = digest(artifact)
    artifact_manifest = json.loads(artifact_manifest_path.read_text())
    required = {
        "artifact_sha256": artifact_hash,
        "artifact_status": "evidence_only_not_deployable",
        "release_policy": "validated-angular-only-v1",
        "n_energy_release_nodes": 0,
        "n_angular_release_nodes": 141,
        "n_fully_suppressed_nodes": 3,
        "stability_pass": True,
        "runtime_load_verified": True,
    }
    for key, expected in required.items():
        if artifact_manifest.get(key) != expected:
            raise SystemExit(
                f"artifact manifest {key}={artifact_manifest.get(key)!r}; "
                f"expected {expected!r}")
    if not artifact_manifest.get("coefficient_file_sha256"):
        raise SystemExit("artifact manifest lacks coefficient provenance")
    released = [row for row in artifact_manifest.get("heldout_support_validation", [])
                if row.get("angular_release")]
    if len(released) != 141 or not all(row.get("pass") for row in released):
        raise SystemExit("the 141 released angular nodes lack held-out support")

    hcs = json.loads(Path(args.hcs_summary).read_text())
    hcs_required = {
        "protocol_version": "hcs-ng-v8",
        "analysis_revision": "hcs-ng-analysis-v3",
        "mode": "sweep",
        "model_variant": "angular_evidence",
        "artifact_sha256": artifact_hash,
        "invariant_corrections": True,
        "orientation_integrator": "symmetric_midpoint_v1",
        "study_campaign_pass": True,
        "n_tasks": 360,
        "n_completed_tasks": 360,
    }
    for key, expected in hcs_required.items():
        if hcs.get(key) != expected:
            raise SystemExit(
                f"HCS-v8 summary {key}={hcs.get(key)!r}; expected {expected!r}")
    if hcs.get("failed_tasks") or hcs.get("missing_tasks"):
        raise SystemExit("HCS-v8 evidence is incomplete")

    provenance_path = Path(args.reference_provenance)
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("role") != "independent_comparison_only_not_closure_calibration":
        raise SystemExit("reference role would permit calibration leakage")
    for path_key, hash_key in (
            ("dem_frozen_path", "dem_frozen_sha256"),
            ("sphere_frozen_path", "sphere_frozen_sha256")):
        frozen = Path(provenance[path_key])
        if not frozen.is_file() or digest(frozen) != provenance[hash_key]:
            raise SystemExit(f"frozen reference failed its checksum: {frozen}")
    if provenance.get("stress_policy", {}).get("primary_rod_comparison") != "kinetic_stress":
        raise SystemExit("reference provenance has an unsafe rod-stress policy")

    designs = (
        (Path(args.numerics_manifest), "numerics", 108),
        (Path(args.pilot_manifest), "pilot", 168),
        (Path(args.full_manifest), "full", 736),
    )
    for path, mode, expected_count in designs:
        rows = manifest_rows(path)
        if len(rows) != expected_count:
            raise SystemExit(f"{mode} design has {len(rows)} rows, expected {expected_count}")
        if {row["mode"] for row in rows} != {mode}:
            raise SystemExit(f"{path} mixes campaign modes")
        if {row.get("protocol_version") for row in rows} != {PROTOCOL}:
            raise SystemExit(f"{mode} does not use {PROTOCOL}")
        if {row.get("artifact_sha256") for row in rows} != {artifact_hash}:
            raise SystemExit(f"{mode} is not bound to the frozen artifact bytes")
        if [int(row["task_id"]) for row in rows] != list(range(len(rows))):
            raise SystemExit(f"{path} task IDs are not contiguous")
        if {int(row["particles"]) for row in rows} != {10_000}:
            raise SystemExit(f"{mode} must use the validated 10,000-particle design")
        if {float(row["volume_fraction"]) for row in rows} != {0.01}:
            raise SystemExit(f"{mode} must preserve phi=0.01")
        if max(float(row["alpha"]) for row in rows) >= 1.0:
            raise SystemExit("alpha=1 cannot be part of unthermostatted steady USF")
        if not all(0.5 <= float(row["alpha"]) <= 0.95
                   and 1.0 <= float(row["aspect_ratio"]) <= 3.0 for row in rows):
            raise SystemExit(f"{mode} contains coordinates outside the declared domain")
        if not all(float(row["evaluation_start_tau"]) ==
                   0.60 * float(row["tau_end"]) for row in rows):
            raise SystemExit(f"{mode} has an inconsistent evaluation window")
    full = manifest_rows(Path(args.full_manifest))
    if sum(row["coordinate_role"] == "interpolation_holdout" for row in full) != 96:
        raise SystemExit("full design lacks the 96 off-grid interpolation runs")
    print(json.dumps({
        "prerequisites_pass": True,
        "protocol": PROTOCOL,
        "artifact_sha256": artifact_hash,
        "reference_provenance_sha256": digest(provenance_path),
        "hcs_v8_pass": True,
        "design_rows": {mode: count for _, mode, count in designs},
    }, indent=2))


if __name__ == "__main__":
    main()
