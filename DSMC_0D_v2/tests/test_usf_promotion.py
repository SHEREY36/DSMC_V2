import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n")


def test_promotion_preserves_bytes_and_binds_both_flow_verdicts(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact = source_dir / "closure_v2.npz"
    artifact.write_bytes(b"small frozen artifact fixture")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    write_json(source_dir / "manifest.json", {
        "artifact_sha256": artifact_hash,
        "artifact_status": "evidence_only_not_deployable",
        "release_policy": "validated-angular-only-v1",
    })
    hcs = tmp_path / "hcs.json"
    write_json(hcs, {
        "protocol_version": "hcs-ng-v8",
        "analysis_revision": "hcs-ng-analysis-v3",
        "study_campaign_pass": True,
        "artifact_sha256": artifact_hash,
        "n_tasks": 360,
        "n_completed_tasks": 360,
    })
    provenance = tmp_path / "provenance.json"
    write_json(provenance, {
        "role": "independent_comparison_only_not_closure_calibration"})
    provenance_hash = hashlib.sha256(provenance.read_bytes()).hexdigest()
    usf = tmp_path / "usf.json"
    write_json(usf, {
        "protocol": "usf-crossflow-v1",
        "analysis_revision": "usf-crossflow-analysis-v2",
        "correction_support_policy": "adaptive_base_law_v1",
        "adaptive_base_law_policy_pass": True,
        "strict_correction_coverage_pass": False,
        "correction_support_case_counts": {
            "correction_supported": 40, "base_law_fallback": 24,
            "mixed_support_base_law_fallback": 6,
        },
        "mode": "full",
        "n_expected": 736, "n_valid": 736, "n_failures": 0,
        "complete": True, "stage_pass": True, "promotion_ready": True,
        "deployment_ready": False,
        "artifact_sha256_values": [artifact_hash],
        "reference_provenance_sha256_values": [provenance_hash],
    })
    output = tmp_path / "promoted"
    subprocess.run([
        sys.executable,
        str(ROOT / "DSMC_0D_v2/scripts/promote_usf_validated_artifact.py"),
        "--artifact", str(artifact), "--hcs-summary", str(hcs),
        "--usf-summary", str(usf), "--reference-provenance", str(provenance),
        "--output-dir", str(output),
    ], check=True, capture_output=True, text=True)
    assert (output / "closure_v2.npz").read_bytes() == artifact.read_bytes()
    promoted = json.loads((output / "manifest.json").read_text())
    assert promoted["deployment_ready"] is True
    assert promoted["artifact_sha256"] == artifact_hash
    assert promoted["source_evidence_artifact_sha256"] == artifact_hash
    assert promoted["artifact_status"] == "deployment_ready_hcs_usf_kinetic_v2"
    assert promoted["release_policy"].endswith("usf-crossflow-v2")
    assert promoted["validated_runtime_policy"]["USF"] == (
        "adaptive_base_law_v1")
    assert promoted["validated_runtime_policy"][
        "USF_strict_correction_coverage_pass"] is False
    assert promoted["validated_domain"]["USF_unthermostatted_steady"]["alpha"] == [
        0.5, 0.95]
