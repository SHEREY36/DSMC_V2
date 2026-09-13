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
    energy_error = {scope: defaultdict(list)
                    for scope in ("central", "heldout")}
    tangent_implementation_error = {scope: []
                                    for scope in ("central", "heldout")}
    angular_error = {scope: defaultdict(list)
                     for scope in ("central", "heldout")}
    node_rows = []
    for key, items in sorted(grouped.items()):
        baseline = json.loads(baseline_paths[key].read_text())
        excited = [item[1] for item in items]
        fitted = fit_correction_coefficients([baseline, *excited])
        beta = np.asarray(fitted["beta"]) * np.asarray(fitted["beta_deployed"])
        center = np.asarray(fitted["feature_center"])
        heldout = [node for node in excited
                   if np.isclose(abs(float(node["excitation"]["eta"])), 0.5)]
        validation_nodes = [node for node in excited
                            if np.isclose(abs(float(node["excitation"]["eta"])), 0.25)
                            or np.isclose(abs(float(node["excitation"]["eta"])), 0.5)]
        node_parameter = defaultdict(list)
        for node in validation_nodes:
            amplitude = abs(float(node["excitation"]["eta"]))
            scope = "central" if np.isclose(amplitude, 0.25) else "heldout"
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
                if scope == "heldout":
                    actual_delta = (float(node[section][name])
                                    - float(baseline[section][name]))
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
                    energy_error[scope][label].append(float(
                        np.mean(np.abs(candidate_q - exact_q))))
                energy_error[scope]["multivariate_exact"].append(float(
                    np.mean(np.abs(predicted_q - exact_q))))
                energy_error[scope]["artifact_tangent"].append(float(
                    np.mean(np.abs(tangent_q - exact_q))))
                tangent_implementation_error[scope].append(float(
                    np.mean(np.abs(tangent_q - predicted_q))))

            exact_angle = angular_quantiles(np.array([
                node["angular"]["eta1"], node["angular"]["eta2"]]), probability)
            for label, candidate in (
                ("baseline", baseline["angular"]),
                ("multivariate", predicted["angular"]),
            ):
                candidate_angle = angular_quantiles(np.array([
                    candidate["eta1"], candidate["eta2"]]), probability)
                angular_error[scope][label].append(float(
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
            "material_parameter_linearity_pass": fitted["linearity_pass"],
            "suppressed_immaterial_parameters": [
                name for name, fit in fitted["parameter_fits"].items()
                if fit.get("immaterial_response_suppressed", False)],
            "elastic_constraints": fitted["elastic_constraints"],
        })

    global_relative = {}
    for name, pairs in parameter_errors.items():
        actual, predicted = np.asarray(pairs).T
        scale = max(float(np.ptp(actual)), float(np.max(np.abs(actual))), 1.0e-12)
        global_relative[name] = float(
            np.sqrt(np.mean((actual - predicted) ** 2)) / scale)
    energy_metrics = {
        scope: {name: _percentiles(value) for name, value in values.items()}
        for scope, values in energy_error.items()}
    angular_metrics = {
        scope: {name: _percentiles(value) for name, value in values.items()}
        for scope, values in angular_error.items()}
    tangent_metrics = {
        scope: _percentiles(values)
        for scope, values in tangent_implementation_error.items()}
    parameter_pass = bool(all(
        node["material_parameter_linearity_pass"] for node in node_rows))
    central_energy = energy_metrics["central"]
    central_angular = angular_metrics["central"]
    heldout_energy = energy_metrics["heldout"]
    heldout_angular = angular_metrics["heldout"]
    heldout_exact_response_pass = bool(
        heldout_energy["multivariate_exact"]["p95"]
        < heldout_energy["baseline"]["p95"]
        and heldout_angular["multivariate"]["p95"]
        < heldout_angular["baseline"]["p95"])
    distribution_pass = bool(
        central_energy["artifact_tangent"]["p95"]
        < central_energy["baseline"]["p95"]
        and central_energy["artifact_tangent"]["p95"]
        < central_energy["lambda1_only"]["p95"]
        and tangent_metrics["central"]["p95"] <= 0.005
        and central_angular["multivariate"]["p95"]
        < central_angular["baseline"]["p95"]
        and heldout_exact_response_pass)
    payload = {
        "manifest": args.manifest,
        "validation_contract": "multivariate-response-offline-v1",
        "parameter_order": list(CORRECTION_PARAMETER_NAMES),
        "n_physical_nodes": len(node_rows),
        "n_heldout_excitations": int(sum(
            np.isclose(abs(float(row["eta"])), 0.5) for row in rows)),
        "node_validation": node_rows,
        "global_heldout_relative_rmse": global_relative,
        # Compatibility keys retain the held-out diagnostics.
        "conditional_energy_quantile_wasserstein1": heldout_energy,
        "artifact_tangent_to_exact_response_wasserstein1": tangent_metrics["heldout"],
        "angular_quantile_wasserstein1": heldout_angular,
        "central_trust_region": {
            "excitation_amplitude": 0.25,
            "conditional_energy_quantile_wasserstein1": central_energy,
            "artifact_tangent_to_exact_response_wasserstein1":
                tangent_metrics["central"],
            "angular_quantile_wasserstein1": central_angular,
        },
        "heldout_exact_response_pass": heldout_exact_response_pass,
        "parameter_response_pass": parameter_pass,
        "distribution_response_pass": distribution_pass,
        "offline_validation_pass": bool(parameter_pass and distribution_pass),
        "scope": ("runtime distribution implementation is released on the fitted "
                  "|eta|=0.25 trust region; both signs of |eta|=0.5 remain held out "
                  "for material natural-parameter linearity and exact-law response; "
                  "conditional energy evaluated at z_in={0.2,0.5,0.8}"),
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
                "central_trust_region": payload["central_trust_region"],
                "heldout_conditional_energy_quantile_wasserstein1": heldout_energy,
                "heldout_artifact_tangent_to_exact_response_wasserstein1":
                    tangent_metrics["heldout"],
                "heldout_angular_quantile_wasserstein1": heldout_angular,
            }, indent=2, sort_keys=True) + "\n```\n")
    print(json.dumps({key: payload[key] for key in (
        "n_physical_nodes", "n_heldout_excitations",
        "parameter_response_pass", "distribution_response_pass",
        "offline_validation_pass")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
