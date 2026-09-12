#!/usr/bin/env python3
"""Audit overlap, rank, linearity, and parameter response of an excitation pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES
from coll_models_v2.response import fit_response


PARAMETERS = (
    ("energy", "p_exch"), ("energy", "lambda1"),
    ("energy", "lambda2"), ("energy", "lambda3"),
    ("energy", "lambda4"), ("energy", "lambda5"),
    ("energy", "lambda6"), ("angular", "eta1"),
    ("angular", "eta2"), ("angular", "rho_z_cosine"),
)
CORRECTION_PARAMETERS = ("lambda1", "lambda2", "lambda3", "lambda4", "eta1", "eta2")


def digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))

    failures = []
    grouped = defaultdict(list)
    for row in rows:
        path = Path(row["output_file"])
        if not path.is_file():
            failures.append({"task_id": int(row["task_id"]), "reason": "missing_output"})
            continue
        result = json.loads(path.read_text())
        if result.get("excitation_status") != "pass":
            failures.append({"task_id": int(row["task_id"]), "reason": "blocked",
                             "detail": result.get("error")})
            continue
        expected = tuple(float(row[name]) for name in
                         ("alpha", "theta", "aspect_ratio"))
        actual = tuple(float(result[name]) for name in
                       ("alpha", "theta", "aspect_ratio"))
        baseline = json.loads(Path(row["baseline_estimate"]).read_text())
        provenance = result.get("excitation_provenance", {})
        if actual != expected or int(result.get("ensemble_id", -1)) != int(row["ensemble_id"]) \
                or result.get("excitation", {}).get("family") != row["family"] \
                or float(result.get("excitation", {}).get("eta", np.nan)) != float(row["eta"]) \
                or provenance.get("baseline_estimate_digest") != digest(baseline):
            failures.append({"task_id": int(row["task_id"]),
                             "reason": "stale_or_mismatched_output"})
            continue
        key = tuple(float(row[name]) for name in ("alpha", "theta", "aspect_ratio"))
        grouped[key].append((row, result))

    nodes = []
    for key, items in sorted(grouped.items()):
        baseline = json.loads(Path(items[0][0]["baseline_estimate"]).read_text())
        mode = items[0][0]["mode"]
        design_names = (("a2_tr", "a2_rot", "a11", "A_cu")
                        if mode == "hcs-pilot" else FEATURE_NAMES)
        base_x = baseline.get("cell_features") or baseline["proposal_features"]
        x = np.array([[result["cell_features"][name] - base_x[name]
                       for name in design_names] for _, result in items])
        scale = np.linalg.norm(x, axis=0)
        scaled = x / np.where(scale > 0.0, scale, 1.0)
        rank = int(np.linalg.matrix_rank(scaled))
        expected_rank = len(design_names)
        response = {}
        for section, name in PARAMETERS:
            if name not in baseline.get(section, {}) \
                    or any(name not in result.get(section, {}) for _, result in items):
                continue
            y = np.array([result[section][name] - baseline[section][name]
                          for _, result in items])
            standardised = []
            base_se = baseline.get("uncertainty", {}).get(name, {}).get("standard_error")
            for (_, result), delta in zip(items, y):
                excited_se = result.get("uncertainty", {}).get(name, {}).get("standard_error")
                if base_se is not None and excited_se is not None:
                    standardised.append(abs(float(delta)) /
                                        max(np.hypot(base_se, excited_se), 1.0e-30))
            fitted = fit_response(baseline, [result for _, result in items],
                                  design_names, section, name)
            fitted.update({
                "maximum_standardized_shift": (max(standardised) if standardised else None),
                "material_response": bool(standardised and max(standardised) >= 3.0),
                "linearity_pass": bool(fitted["validation_relative_rmse"] is not None
                                       and fitted["validation_relative_rmse"] <= 0.15),
            })
            response[name] = fitted
        ess = [result["excitation"]["ess_fraction"] for _, result in items]
        share = [result["excitation"]["max_weight_share"] for _, result in items]
        precision = [result.get("qa", {}).get("precision_pass", False)
                     for _, result in items]
        sentinel = [result.get("qa", {}).get("sentinel_pass", False)
                    for _, result in items]
        central = [np.isclose(abs(float(row["eta"])), 0.25)
                   for row, _ in items]
        heldout = [np.isclose(abs(float(row["eta"])), 0.50)
                   for row, _ in items]
        heldout_acceptable = []
        for (_, result), selected in zip(items, heldout):
            if not selected:
                continue
            qa = result.get("qa", {})
            model_form_only = (
                set(qa.get("continuation_reasons", [])) == {"model_form"}
                and all(qa.get(name, False) for name in (
                    "angular_projection_pass", "elastic_pass",
                    "energy_projection_pass", "ess_pass",
                    "incoming_partition_pass", "memory_diagnostic_pass",
                    "propensity_pass", "proposal_balance_pass")))
            heldout_acceptable.append(bool(qa.get("sentinel_pass", False)
                                           or model_form_only))
        node = {
            "coordinates": list(key), "mode": mode, "n_excitations": len(items),
            "design_features": list(design_names), "design_rank": rank,
            "expected_rank": expected_rank,
            "condition_number_scaled": float(np.linalg.cond(scaled)),
            "minimum_ess_fraction": float(min(ess)),
            "maximum_weight_share": float(max(share)),
            "all_sentinel_pass": bool(all(sentinel)),
            "training_sentinel_pass": bool(all(
                value for value, selected in zip(sentinel, central) if selected)),
            "heldout_sentinel_pass": bool(all(
                value for value, selected in zip(sentinel, heldout) if selected)),
            "n_heldout_sentinel_failures": int(sum(
                not value for value, selected in zip(sentinel, heldout) if selected)),
            "heldout_boundary_acceptable": bool(all(heldout_acceptable)),
            "all_pointwise_precision_pass": bool(all(precision)),
            # Compatibility alias. Pointwise precision is reported, but it is
            # no longer confused with a response-model release decision.
            "all_precision_pass": bool(all(precision)),
            "parameter_response": response,
        }
        node["screening_pass"] = bool(
            rank == expected_rank and min(ess) >= 0.5 and max(share) <= 0.01
            and node["training_sentinel_pass"])
        required = [response.get(name, {}) for name in CORRECTION_PARAMETERS]
        node["response_linearity_pass"] = bool(
            all(item.get("linearity_pass", False) for item in required))
        node["response_fit_ready"] = bool(
            node["screening_pass"] and node["response_linearity_pass"]
            and node["heldout_boundary_acceptable"]
            and node["condition_number_scaled"] <= 100.0)
        nodes.append(node)

    screening_pass = bool(not failures and nodes
                          and all(node["screening_pass"] for node in nodes))
    response_fit_ready = bool(not failures and nodes
                              and all(node["response_fit_ready"] for node in nodes))
    mode = rows[0]["mode"] if rows else None
    heldout_sentinel_pass = bool(nodes and all(
        node["heldout_sentinel_pass"] for node in nodes))
    candidate_artifact_ready = bool(
        mode in ("correction-grid", "usf-extension") and response_fit_ready)
    blockers = ["pilot_does_not_modify_artifact",
                "corrected_dynamics_not_validated",
                "independent_direct_ctc_validation_missing"]
    if mode == "hcs-pilot":
        blockers.insert(1, "full_14_feature_basis_not_sampled")
    payload = {
        "manifest": args.manifest, "n_tasks": len(rows), "n_failures": len(failures),
        "failures": failures, "nodes": nodes,
        "screening_pass": screening_pass,
        "training_screening_pass": screening_pass,
        "heldout_sentinel_pass": heldout_sentinel_pass,
        "restricted_to_calibrated_feature_domain": not heldout_sentinel_pass,
        "pointwise_precision_pass": bool(nodes and all(
            node["all_pointwise_precision_pass"] for node in nodes)),
        "response_fit_ready": response_fit_ready,
        "candidate_artifact_ready": candidate_artifact_ready,
        # Deployment needs a rebuilt candidate artifact followed by corrected
        # HCS and direct-CTC validation. A pilot can never satisfy those gates.
        "deployment_ready": False,
        "deployment_blockers": blockers,
        "decision": ("build a correction-enabled candidate when candidate_artifact_ready "
                     "is true; held-out boundary failures restrict the calibrated feature "
                     "domain but do not invalidate a full-rank central response fit"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in
                      ("n_tasks", "n_failures", "screening_pass",
                       "heldout_sentinel_pass", "response_fit_ready",
                       "candidate_artifact_ready", "deployment_ready")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
