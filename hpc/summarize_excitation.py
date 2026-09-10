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


PARAMETERS = (
    ("energy", "p_exch"), ("energy", "lambda1"),
    ("energy", "lambda2"), ("energy", "lambda3"),
    ("energy", "lambda4"), ("energy", "lambda5"),
    ("energy", "lambda6"), ("angular", "eta1"),
    ("angular", "eta2"), ("angular", "rho_z_cosine"),
)


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
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            residual = y - x @ beta
            scale_y = max(float(np.ptp(y)), 1.0e-12)
            standardised = []
            base_se = baseline.get("uncertainty", {}).get(name, {}).get("standard_error")
            for (_, result), delta in zip(items, y):
                excited_se = result.get("uncertainty", {}).get(name, {}).get("standard_error")
                if base_se is not None and excited_se is not None:
                    standardised.append(abs(float(delta)) /
                                        max(np.hypot(base_se, excited_se), 1.0e-30))
            response[name] = {
                "coefficients": dict(zip(design_names, beta.tolist())),
                "relative_linear_rmse": float(np.sqrt(np.mean(residual ** 2)) / scale_y),
                "maximum_standardized_shift": (max(standardised) if standardised else None),
                "material_response": bool(standardised and max(standardised) >= 3.0),
            }
        ess = [result["excitation"]["ess_fraction"] for _, result in items]
        share = [result["excitation"]["max_weight_share"] for _, result in items]
        precision = [result.get("qa", {}).get("precision_pass", False)
                     for _, result in items]
        sentinel = [result.get("qa", {}).get("sentinel_pass", False)
                    for _, result in items]
        node = {
            "coordinates": list(key), "mode": mode, "n_excitations": len(items),
            "design_features": list(design_names), "design_rank": rank,
            "expected_rank": expected_rank,
            "condition_number_scaled": float(np.linalg.cond(scaled)),
            "minimum_ess_fraction": float(min(ess)),
            "maximum_weight_share": float(max(share)),
            "all_sentinel_pass": bool(all(sentinel)),
            "all_precision_pass": bool(all(precision)),
            "parameter_response": response,
        }
        node["screening_pass"] = bool(
            rank == expected_rank and min(ess) >= 0.5 and max(share) <= 0.01
            and all(sentinel))
        nodes.append(node)

    payload = {
        "manifest": args.manifest, "n_tasks": len(rows), "n_failures": len(failures),
        "failures": failures, "nodes": nodes,
        "screening_pass": bool(not failures and nodes
                               and all(node["screening_pass"] for node in nodes)),
        "deployment_ready": bool(not failures and nodes
                                  and all(node["screening_pass"]
                                          and node["all_precision_pass"] for node in nodes)),
        "decision": ("inspect parameter_response before choosing the deployed "
                     "natural-parameter correction; this pilot never modifies the artifact"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in
                      ("n_tasks", "n_failures", "screening_pass", "deployment_ready")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
