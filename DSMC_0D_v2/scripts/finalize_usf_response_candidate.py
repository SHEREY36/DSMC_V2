#!/usr/bin/env python3
"""Promote a nonlinear candidate only after direct CTC, HCS, and USF gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--direct-report", required=True)
    parser.add_argument("--hcs-summary", required=True)
    parser.add_argument("--usf-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = Path(args.artifact)
    direct = json.loads(Path(args.direct_report).read_text())
    hcs = json.loads(Path(args.hcs_summary).read_text())
    usf = json.loads(Path(args.usf_summary).read_text())
    artifact_hash = digest(artifact)
    reasons = []
    if sorted(direct.get("energy_release_alphas", [])) != [0.5, 0.8, 0.95]:
        reasons.append("direct_energy_response_not_released_at_all_three_alphas")
    if sorted(direct.get("angular_release_alphas", [])) != [0.5, 0.8, 0.95]:
        reasons.append("direct_angular_response_not_released_at_all_three_alphas")
    if not hcs.get("full_domain_physics_gate_pass", False):
        reasons.append("full_domain_hcs_physics_gate_failed")
    if not hcs.get("production_gate_pass", False):
        reasons.append("hcs_production_gate_failed")
    if hcs.get("artifact_sha256") != artifact_hash:
        reasons.append("hcs_artifact_hash_mismatch")
    if not usf.get("stage_pass", False):
        reasons.append("paired_usf_response_validation_failed")
    if not usf.get("strict_correction_coverage_pass", False):
        reasons.append("usf_correction_coverage_failed")
    if usf.get("artifact_sha256_values") != [artifact_hash]:
        reasons.append("usf_artifact_hash_mismatch")
    manifest_path = artifact.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("correction_digest") != direct.get(
            "coefficient_rows_sha256"):
        reasons.append("direct_response_coefficients_do_not_match_artifact")
    if manifest.get("artifact_status") \
            != "candidate_not_deployable_pending_hcs_usf_validation":
        reasons.append("artifact_is_not_pending_nonlinear_candidate")
    passed = not reasons
    decision = {
        "validation_contract": "usf-nonlinear-response-promotion-v1",
        "pass": passed, "reasons": reasons,
        "artifact": str(artifact), "artifact_sha256": artifact_hash,
        "direct_report": str(Path(args.direct_report)),
        "hcs_summary": str(Path(args.hcs_summary)),
        "usf_summary": str(Path(args.usf_summary)),
        "capability": {
            "sphere": "AR=1 exact hard-sphere branch",
            "base_law_rods": "1.1<=AR<=3, 0.5<=alpha<=1",
            "USF_nonlinear_correction_validated": (
                "1.1<=AR<=3 at alpha=0.5,0.8,0.95; all AR-cell midpoints "
                "validated at alpha=0.65,0.875; continuous interpolation "
                "remains subject to runtime invariant support"),
            "DEM_role": (
                "external validation/guidance only; never a fit target; "
                "rod-USF DEM is not a release gate"),
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("candidate promotion blocked: " + ", ".join(reasons))
    manifest.update({
        "artifact_status": "deployment_ready_hcs_usf_internal_v1",
        "release_policy": "direct_ctc_validated_additive_quadratic_v1",
        "artifact_sha256": artifact_hash,
        "promotion_decision": str(output),
        "validated_capability": decision["capability"],
        "dem_used_as_fit_target": False,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
