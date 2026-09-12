#!/usr/bin/env python3
"""Compare fresh-CTC excitation responses with the deployed artifact Jacobian."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from dsmc_v2.artifact import VariationalClosure
from coll_models_v2.excitation import EXCITATION_FAMILIES
from dsmc_v2_contracts import FEATURE_NAMES


PARAMETERS = ("lambda1", "lambda2", "lambda3", "lambda4", "eta1", "eta2")


def digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def parameter_vector(node: dict) -> np.ndarray:
    return np.array([
        *(float(node["energy"][name]) for name in PARAMETERS[:4]),
        *(float(node["angular"][name]) for name in PARAMETERS[4:]),
    ])


def standard_errors(node: dict) -> np.ndarray:
    uncertainty = node.get("uncertainty", {})
    return np.array([
        float(uncertainty.get(name, {}).get("standard_error", np.nan))
        for name in PARAMETERS
    ])


def json_parameter_map(values: np.ndarray) -> dict:
    return {name: (float(value) if np.isfinite(value) else None)
            for name, value in zip(PARAMETERS, values)}


def response_scores(observed: np.ndarray, predicted: np.ndarray,
                    uncertainty: np.ndarray) -> dict:
    """Score one response parameter over all independent holdout directions."""
    residual = predicted - observed
    scale = max(float(np.ptp(observed)), float(np.max(np.abs(observed))), 1.0e-12)
    finite_uncertainty = np.isfinite(uncertainty) & (uncertainty > 0.0)
    return {
        "observed_response_min": float(np.min(observed)),
        "observed_response_max": float(np.max(observed)),
        "response_scale": scale,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "relative_rmse": float(np.sqrt(np.mean(residual * residual)) / scale),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
        "correlation": (float(np.corrcoef(observed, predicted)[0, 1])
                        if np.std(observed) > 0.0 and np.std(predicted) > 0.0
                        else None),
        "median_absolute_standardized_residual": (
            float(np.median(np.abs(residual[finite_uncertainty])
                            / uncertainty[finite_uncertainty]))
            if np.any(finite_uncertainty) else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-estimate", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--relative-rmse-max", type=float, default=0.20)
    args = parser.parse_args()

    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    baseline = json.loads(Path(args.baseline_estimate).read_text())
    baseline_digest = digest(baseline)
    physical = np.array([baseline[name] for name in
                         ("alpha", "theta", "aspect_ratio")], dtype=float)
    base_features = np.array([baseline["cell_features"][name]
                              for name in FEATURE_NAMES], dtype=float)
    base_parameters = parameter_vector(baseline)
    base_se = standard_errors(baseline)
    closure = VariationalClosure(args.artifact, corrections_enabled=True)
    base_state = closure.kernel_state(*physical, base_features)
    base_correction = np.array([
        base_state["parameter_correction"].get(name, 0.0) for name in PARAMETERS
    ])

    failures, observations = [], []
    seen = set()
    for row in rows:
        path = Path(row["output_file"])
        if not path.is_file():
            failures.append({"task_id": int(row["task_id"]), "reason": "missing_output"})
            continue
        node = json.loads(path.read_text())
        key = (row["family"], float(row["eta"]))
        if key in seen:
            failures.append({"task_id": int(row["task_id"]), "reason": "duplicate_design"})
            continue
        seen.add(key)
        provenance = node.get("excitation_provenance", {})
        node_physical = np.array([node.get(name, np.nan) for name in
                                  ("alpha", "theta", "aspect_ratio")], dtype=float)
        excitation = node.get("excitation", {})
        if node.get("excitation_status") != "pass" \
                or not bool(excitation.get("usable", False)) \
                or not np.array_equal(node_physical, physical) \
                or excitation.get("family") != row["family"] \
                or float(excitation.get("eta", np.nan)) != float(row["eta"]) \
                or provenance.get("baseline_estimate_digest") != baseline_digest:
            failures.append({"task_id": int(row["task_id"]),
                             "reason": "blocked_or_mismatched_output"})
            continue
        features = np.array([node["cell_features"][name]
                             for name in FEATURE_NAMES], dtype=float)
        state = closure.kernel_state(*physical, features)
        correction = np.array([
            state["parameter_correction"].get(name, 0.0) for name in PARAMETERS
        ])
        observed = parameter_vector(node) - base_parameters
        predicted = correction - base_correction
        # The excited and baseline fits share the baseline shard, hence this
        # quadrature uncertainty is conservative (their covariance is positive).
        uncertainty = np.hypot(standard_errors(node), base_se)
        observations.append({
            "task_id": int(row["task_id"]), "family": row["family"],
            "eta": float(row["eta"]), "ess_fraction": excitation["ess_fraction"],
            "max_weight_share": excitation["max_weight_share"],
            "observed": observed, "predicted": predicted,
            "uncertainty": uncertainty,
        })

    expected = 2 * len(EXCITATION_FAMILIES)
    if len(rows) != expected:
        failures.append({"reason": "wrong_design_size", "expected": expected,
                         "actual": len(rows)})
    if observations:
        observed = np.array([item["observed"] for item in observations])
        predicted = np.array([item["predicted"] for item in observations])
        uncertainty = np.array([item["uncertainty"] for item in observations])
        scores = {name: response_scores(observed[:, index], predicted[:, index],
                                        uncertainty[:, index])
                  for index, name in enumerate(PARAMETERS)}
    else:
        scores = {}
    threshold = float(args.relative_rmse_max)
    response_pass = bool(scores and all(
        score["relative_rmse"] <= threshold for score in scores.values()))
    payload = {
        "schema_version": "independent_ctc_holdout_v1",
        "physical_node": physical.tolist(),
        "artifact": str(Path(args.artifact)),
        "artifact_sha256": file_digest(args.artifact),
        "baseline_estimate": str(Path(args.baseline_estimate)),
        "baseline_source_runs": baseline.get("source_runs", []),
        "n_expected": expected, "n_valid": len(observations),
        "n_failures": len(failures), "failures": failures,
        "minimum_ess_fraction": (min(item["ess_fraction"] for item in observations)
                                 if observations else None),
        "maximum_weight_share": (max(item["max_weight_share"] for item in observations)
                                 if observations else None),
        "relative_rmse_maximum_allowed": threshold,
        "parameter_response": scores,
        "observations": [{
            "task_id": item["task_id"], "family": item["family"],
            "eta": item["eta"], "ess_fraction": item["ess_fraction"],
            "max_weight_share": item["max_weight_share"],
            "observed_response": json_parameter_map(item["observed"]),
            "predicted_response": json_parameter_map(item["predicted"]),
            "conservative_standard_error": json_parameter_map(
                item["uncertainty"]),
        } for item in observations],
        "response_pass": response_pass,
        "validation_pass": bool(not failures and len(observations) == expected
                                and response_pass),
        "interpretation": (
            "New CTC trajectories provide an independent collision/outcome sample. "
            "Boundary-amplitude one-particle importance tilts are evaluated only on "
            "that holdout shard and compared with the frozen artifact Jacobian."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "n_expected", "n_valid", "n_failures", "minimum_ess_fraction",
        "maximum_weight_share", "response_pass", "validation_pass")},
        indent=2, sort_keys=True))
    if not payload["validation_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
