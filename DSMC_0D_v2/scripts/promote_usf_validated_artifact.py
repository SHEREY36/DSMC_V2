#!/usr/bin/env python3
"""Promote unchanged closure bytes after complete HCS and USF validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--hcs-summary", required=True)
    parser.add_argument("--usf-summary", required=True)
    parser.add_argument("--reference-provenance", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.artifact)
    source_manifest_path = source.with_name("manifest.json")
    source_hash = digest(source)
    source_manifest = json.loads(source_manifest_path.read_text())
    if (source_manifest.get("artifact_sha256") != source_hash
            or source_manifest.get("artifact_status")
            != "evidence_only_not_deployable"
            or source_manifest.get("release_policy")
            != "validated-angular-only-v1"):
        raise SystemExit("source is not the frozen angular-evidence artifact")

    hcs_path, usf_path = Path(args.hcs_summary), Path(args.usf_summary)
    hcs, usf = json.loads(hcs_path.read_text()), json.loads(usf_path.read_text())
    if not (hcs.get("protocol_version") == "hcs-ng-v8"
            and hcs.get("analysis_revision") == "hcs-ng-analysis-v3"
            and hcs.get("study_campaign_pass") is True
            and hcs.get("artifact_sha256") == source_hash
            and hcs.get("n_completed_tasks") == hcs.get("n_tasks") == 360):
        raise SystemExit("complete passing HCS-v8 evidence is required")
    if not (usf.get("protocol") == "usf-crossflow-v1"
            and usf.get("mode") == "full"
            and usf.get("n_valid") == usf.get("n_expected") == 736
            and usf.get("n_failures") == 0 and usf.get("complete") is True
            and usf.get("stage_pass") is True
            and usf.get("promotion_ready") is True
            and usf.get("deployment_ready") is False
            and usf.get("artifact_sha256_values") == [source_hash]):
        raise SystemExit("complete passing full-domain USF evidence is required")

    provenance_path = Path(args.reference_provenance)
    provenance = json.loads(provenance_path.read_text())
    expected_provenance_hash = digest(provenance_path)
    if (usf.get("reference_provenance_sha256_values")
            != [expected_provenance_hash]
            or provenance.get("role")
            != "independent_comparison_only_not_closure_calibration"):
        raise SystemExit("USF result does not bind the frozen comparison provenance")

    destination = Path(args.output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty promotion directory {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    artifact_destination = destination / "closure_v2.npz"
    temporary = destination / ".closure_v2.npz.tmp"
    shutil.copyfile(source, temporary)
    if digest(temporary) != source_hash:
        raise SystemExit("promoted artifact copy failed its checksum")
    os.replace(temporary, artifact_destination)

    manifest = dict(source_manifest)
    manifest.update({
        "artifact_status": "deployment_ready_hcs_usf_kinetic_v1",
        "deployment_ready": True,
        "release_policy": "validated-angular-adaptive-hcs-v8-usf-crossflow-v1",
        "source_evidence_artifact": str(source),
        "source_evidence_artifact_sha256": source_hash,
        "artifact_sha256": source_hash,
        "validated_domain": {
            "HCS": {"alpha": [0.5, 1.0], "aspect_ratio": [1.0, 3.0]},
            "USF_unthermostatted_steady": {
                "alpha": [0.5, 0.95], "aspect_ratio": [1.0, 3.0]},
            "USF_alpha_1_exclusion": (
                "elastic unthermostatted USF has no finite steady state"),
        },
        "validated_observables": {
            "HCS": ["temperature_ratio", "non_gaussian_moments", "tails"],
            "USF": ["Tstar", "temperature_ratio", "kinetic_stress",
                    "nematic_order", "stationarity", "energy_balance"],
        },
        "limitations": [
            "DEM is an independent comparison only and was not a calibration gate",
            "rod collisional/total stress remains diagnostic because the exact branch vector is absent",
            "released angular correction is used only inside held-out-supported regions; runtime falls back to the base law elsewhere",
            "energy-response corrections remain suppressed",
        ],
        "validation_evidence": {
            "hcs_summary": str(hcs_path),
            "hcs_summary_sha256": digest(hcs_path),
            "usf_summary": str(usf_path),
            "usf_summary_sha256": digest(usf_path),
            "reference_provenance": str(provenance_path),
            "reference_provenance_sha256": expected_provenance_hash,
        },
    })
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "deployment_ready": True,
        "artifact": str(artifact_destination),
        "artifact_sha256": source_hash,
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
