#!/usr/bin/env python3
"""Offline validation of the multivariate excitation response model.

This consumes existing virtual-excitation JSON files only. It neither reruns
CTC nor modifies an artifact. Central amplitudes fit the response; both signs
of the held-out amplitudes exercise parameter prediction and complete energy/angular
probability laws before an expensive candidate-artifact build is submitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from coll_models_v2.fit_coefficients import (
    CORRECTION_PARAMETERS,
    CORRECTION_PARAMETER_NAMES,
    LINEARITY_TOLERANCE,
    fit_correction_coefficients,
)
from coll_models_v2.projections import angular_quantiles, energy_quantile_table
from dsmc_v2_contracts import FEATURE_NAMES


def _quantiles(energy: dict, z_in: float, loss: float,
               probability: np.ndarray) -> np.ndarray:
    form = energy.get("kernel_form", "conditional_iprojection_v2")
    a = (float(energy["lambda1"]) + float(energy["lambda3"]) * z_in
         + float(energy["lambda4"]) * loss)
    return energy_quantile_table(
        float(energy["lambda3"]), float(energy["lambda2"]), np.array([a]),
        probability, kernel_form=form,
        anchor=(energy.get("anchor_c1", 0.0), energy.get("anchor_c2", 0.0)),
    )[0]


def _percentiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _tangent_quantiles(baseline: dict, predicted: dict, z_in: float,
                       loss: float, probability: np.ndarray) -> np.ndarray:
    """Emulate the candidate artifact's logit-quantile tangent update."""
    form = baseline.get("kernel_form", "conditional_iprojection_v2")
    anchor = (baseline.get("anchor_c1", 0.0), baseline.get("anchor_c2", 0.0))
    a = (float(predicted["lambda1"]) + float(predicted["lambda3"]) * z_in
         + float(predicted["lambda4"]) * loss)

    def table(memory: float, lambda2: float) -> np.ndarray:
        return energy_quantile_table(
            memory, lambda2, np.array([a]), probability,
            kernel_form=form, anchor=anchor)[0]

    base_memory = float(baseline["lambda3"])
    base_lambda2 = float(baseline["lambda2"])
    base = table(base_memory, base_lambda2)
    sensitivities = []
    for name in ("lambda2", "lambda3"):
        value = float(baseline[name])
        step = max(1.0e-4, 1.0e-3 * (1.0 + abs(value)))
        low = table(base_memory - (step if name == "lambda3" else 0.0),
                    base_lambda2 - (step if name == "lambda2" else 0.0))
        high = table(base_memory + (step if name == "lambda3" else 0.0),
                     base_lambda2 + (step if name == "lambda2" else 0.0))
        eps = 1.0e-8
        low = np.clip(low, eps, 1.0 - eps)
        high = np.clip(high, eps, 1.0 - eps)
        sensitivities.append(
            (np.log(high / (1.0 - high)) - np.log(low / (1.0 - low)))
            / (2.0 * step))
    q = np.clip(base, 1.0e-8, 1.0 - 1.0e-8)
    logit = np.log(q / (1.0 - q))
    logit += ((float(predicted["lambda2"]) - base_lambda2) * sensitivities[0]
              + (float(predicted["lambda3"]) - base_memory) * sensitivities[1])
    result = 1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0)))
    result[0], result[-1] = 0.0, 1.0
    return np.maximum.accumulate(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    baseline_paths = {}
    missing = []
    for row in rows:
        path = Path(row["output_file"])
        if not path.is_file():
            missing.append(str(path))
            continue
        key = tuple(float(row[name]) for name in
                    ("alpha", "theta", "aspect_ratio"))
        grouped[key].append((row, json.loads(path.read_text())))
        baseline_paths[key] = Path(row["baseline_estimate"])
    if missing:
        raise SystemExit(f"{len(missing)} excitation outputs are missing")

    probability = np.linspace(0.0, 1.0, 65)
    parameter_errors = defaultdict(list)
    energy_error = defaultdict(list)
    tangent_implementation_error = []
    angular_error = defaultdict(list)
    node_rows = []
    for key, items in sorted(grouped.items()):
        baseline = json.loads(baseline_paths[key].read_text())
        excited = [item[1] for item in items]
        fitted = fit_correction_coefficients([baseline, *excited])
        beta = np.asarray(fitted["beta"]) * np.asarray(fitted["beta_deployed"])
        center = np.asarray(fitted["feature_center"])
        heldout = [node for node in excited
                   if np.isclose(abs(float(node["excitation"]["eta"])), 0.5)]
        node_parameter = defaultdict(list)
        for node in heldout:
            features = np.asarray([node["cell_features"][name]
                                   for name in FEATURE_NAMES])
            delta = beta @ (features - center)
            predicted = {
                "energy": dict(baseline["energy"]),
                "angular": dict(baseline["angular"]),
            }
            lambda1_only = {
                "energy": dict(baseline["energy"]),
                "angular": dict(baseline["angular"]),
            }
            for index, (section, name) in enumerate(CORRECTION_PARAMETERS):
                predicted[section][name] = float(baseline[section][name]) + delta[index]
                actual_delta = float(node[section][name]) - float(baseline[section][name])
                node_parameter[name].append((actual_delta, delta[index]))
                parameter_errors[name].append((actual_delta, delta[index]))
            lambda1_only["energy"]["lambda1"] += delta[0]

            loss = float(node["energy"].get("mean_fractional_loss", 0.0))
            for z_in in (0.2, 0.5, 0.8):
                exact_q = _quantiles(node["energy"], z_in, loss, probability)
                predicted_q = _quantiles(
                    predicted["energy"], z_in, loss, probability)
                tangent_q = _tangent_quantiles(
                    baseline["energy"], predicted["energy"], z_in, loss,
                    probability)
                for label, candidate in (
                    ("baseline", baseline["energy"]),
                    ("lambda1_only", lambda1_only["energy"]),
                ):
                    candidate_q = _quantiles(candidate, z_in, loss, probability)
                    energy_error[label].append(float(np.mean(np.abs(candidate_q - exact_q))))
                energy_error["multivariate_exact"].append(float(
                    np.mean(np.abs(predicted_q - exact_q))))
                energy_error["artifact_tangent"].append(float(
                    np.mean(np.abs(tangent_q - exact_q))))
                tangent_implementation_error.append(float(
                    np.mean(np.abs(tangent_q - predicted_q))))

            exact_angle = angular_quantiles(np.array([
                node["angular"]["eta1"], node["angular"]["eta2"]]), probability)
            for label, candidate in (
                ("baseline", baseline["angular"]),
                ("multivariate", predicted["angular"]),
            ):
                candidate_angle = angular_quantiles(np.array([
                    candidate["eta1"], candidate["eta2"]]), probability)
                angular_error[label].append(float(
                    np.mean(np.abs(candidate_angle - exact_angle))))

        relative = {}
        for name, pairs in node_parameter.items():
            actual, predicted = np.asarray(pairs).T
            scale = max(float(np.ptp(actual)), float(np.max(np.abs(actual))), 1.0e-12)
            relative[name] = float(np.sqrt(np.mean((actual - predicted) ** 2)) / scale)
        node_rows.append({
            "coordinates": list(key),
            "design_rank": fitted["design_rank"],
            "condition_number_scaled": fitted["condition_number"],
            "heldout_positive_relative_rmse": relative,
            "maximum_positive_subset_relative_rmse": max(relative.values()),
            "maximum_full_heldout_relative_rmse":
                fitted["maximum_validation_relative_rmse"],
            "elastic_constraints": fitted["elastic_constraints"],
        })

    global_relative = {}
    for name, pairs in parameter_errors.items():
        actual, predicted = np.asarray(pairs).T
        scale = max(float(np.ptp(actual)), float(np.max(np.abs(actual))), 1.0e-12)
        global_relative[name] = float(
            np.sqrt(np.mean((actual - predicted) ** 2)) / scale)
    energy_metrics = {name: _percentiles(value) for name, value in energy_error.items()}
    angular_metrics = {name: _percentiles(value) for name, value in angular_error.items()}
    tangent_metrics = _percentiles(tangent_implementation_error)
    parameter_pass = bool(all(
        node["maximum_full_heldout_relative_rmse"] <= LINEARITY_TOLERANCE
        for node in node_rows))
    distribution_pass = bool(
        energy_metrics["artifact_tangent"]["p95"] < energy_metrics["baseline"]["p95"]
        and energy_metrics["artifact_tangent"]["p95"]
        < energy_metrics["lambda1_only"]["p95"]
        and tangent_metrics["p95"] <= 0.005
        and angular_metrics["multivariate"]["p95"] < angular_metrics["baseline"]["p95"])
    payload = {
        "manifest": args.manifest,
        "validation_contract": "multivariate-response-offline-v1",
        "parameter_order": list(CORRECTION_PARAMETER_NAMES),
        "n_physical_nodes": len(node_rows),
        "n_heldout_excitations": int(sum(
            np.isclose(abs(float(row["eta"])), 0.5) for row in rows)),
        "node_validation": node_rows,
        "global_heldout_relative_rmse": global_relative,
        "conditional_energy_quantile_wasserstein1": energy_metrics,
        "artifact_tangent_to_exact_response_wasserstein1": tangent_metrics,
        "angular_quantile_wasserstein1": angular_metrics,
        "parameter_response_pass": parameter_pass,
        "distribution_response_pass": distribution_pass,
        "offline_validation_pass": bool(parameter_pass and distribution_pass),
        "scope": ("both signs of |eta|=0.5 held-out excitations; conditional energy "
                  "evaluated at z_in={0.2,0.5,0.8}"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Multivariate excitation repair: offline validation\n\n"
            f"- Physical nodes: {payload['n_physical_nodes']}\n"
            f"- Held-out excitations: {payload['n_heldout_excitations']}\n"
            f"- Parameter response pass: `{parameter_pass}`\n"
            f"- Distribution response pass: `{distribution_pass}`\n"
            f"- Offline validation pass: `{payload['offline_validation_pass']}`\n\n"
            "The model fits the central ±0.25 perturbations and validates on both "
            "signs of the |eta|=0.5 boundary without reusing those points for fitting. "
            "The comparison "
            "below is distribution-level: lower Wasserstein-1 distance is better.\n\n"
            "```json\n" + json.dumps({
                "global_heldout_relative_rmse": global_relative,
                "conditional_energy_quantile_wasserstein1": energy_metrics,
                "artifact_tangent_to_exact_response_wasserstein1": tangent_metrics,
                "angular_quantile_wasserstein1": angular_metrics,
            }, indent=2, sort_keys=True) + "\n```\n")
    print(json.dumps({key: payload[key] for key in (
        "n_physical_nodes", "n_heldout_excitations",
        "parameter_response_pass", "distribution_response_pass",
        "offline_validation_pass")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
