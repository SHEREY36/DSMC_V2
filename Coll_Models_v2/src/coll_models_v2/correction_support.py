"""Independent distribution-level validation of correction support.

The response tensor is fitted only on the central excitation amplitude.  A
larger feature hull is released at a physical node only when both signs of the
existing held-out excitation reproduce the exact conditional laws there.
"""

from __future__ import annotations

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES

from .fit_coefficients import CORRECTION_PARAMETERS
from .projections import angular_quantiles, energy_quantile_table
from .fit_exchange import LOGIT_CUBIC_KERNEL, logit_memory_basis
from .excitation import EXCITATION_FAMILIES


HELDOUT_AMPLITUDE = 0.5
TANGENT_MAXIMUM_TOLERANCE = 5.0e-3


def _memory_shift(energy: dict, z_in: float) -> float:
    if energy.get("kernel_form") == LOGIT_CUBIC_KERNEL:
        coefficients = np.asarray(energy["memory_coefficients"], dtype=float).copy()
        coefficients[0] = float(energy["lambda3"])
        basis = logit_memory_basis(
            np.array([z_in]), float(energy["memory_center"]),
            float(energy["memory_scale"]), degree=len(coefficients))
        return float(basis[0] @ coefficients)
    return float(energy["lambda3"]) * z_in


def energy_quantiles(energy: dict, z_in: float, loss: float,
                     probability: np.ndarray) -> np.ndarray:
    form = energy.get("kernel_form", "conditional_iprojection_v2")
    a = (float(energy["lambda1"]) + _memory_shift(energy, z_in)
         + float(energy["lambda4"]) * loss)
    return energy_quantile_table(
        float(energy["lambda3"]), float(energy["lambda2"]), np.array([a]),
        probability, kernel_form=form,
        anchor=(energy.get("anchor_c1", 0.0), energy.get("anchor_c2", 0.0)),
    )[0]


def tangent_quantiles(baseline: dict, predicted: dict, z_in: float,
                      loss: float, probability: np.ndarray) -> np.ndarray:
    """Emulate the deployed logit-quantile tangent update."""
    form = baseline.get("kernel_form", "conditional_iprojection_v2")
    anchor = (baseline.get("anchor_c1", 0.0), baseline.get("anchor_c2", 0.0))
    a = (float(predicted["lambda1"]) + _memory_shift(predicted, z_in)
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
        low, high = np.clip(low, eps, 1.0 - eps), np.clip(high, eps, 1.0 - eps)
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


def _percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def assess_heldout_support(baseline: dict, excited: list[dict], fitted: dict,
                           amplitude: float = HELDOUT_AMPLITUDE) -> dict:
    """Decide whether held-out observations may enlarge one local feature hull."""
    heldout = [node for node in excited
               if node.get("excitation") is not None
               and np.isclose(abs(float(node["excitation"]["eta"])), amplitude)]
    expected = 2 * len(EXCITATION_FAMILIES)
    if len(heldout) != expected:
        return {"pass": False, "amplitude": amplitude,
                "n_observations": len(heldout), "n_expected": expected,
                "reason": "incomplete_heldout_design"}

    probability = np.linspace(0.0, 1.0, 65)
    beta = (np.asarray(fitted["beta"], dtype=float)
            * np.asarray(fitted["beta_deployed"], dtype=bool))
    center = np.asarray(fitted["feature_center"], dtype=float)
    baseline_energy, corrected_energy, exact_energy, tangent_error = [], [], [], []
    baseline_angle, corrected_angle = [], []
    for node in heldout:
        features = np.asarray([node["cell_features"][name] for name in FEATURE_NAMES])
        delta = beta @ (features - center)
        predicted = {"energy": dict(baseline["energy"]),
                     "angular": dict(baseline["angular"])}
        for index, (section, name) in enumerate(CORRECTION_PARAMETERS):
            predicted[section][name] = float(baseline[section][name]) + delta[index]
        if predicted["energy"].get("kernel_form") == LOGIT_CUBIC_KERNEL:
            coefficients = list(predicted["energy"]["memory_coefficients"])
            coefficients[0] = predicted["energy"]["lambda3"]
            predicted["energy"]["memory_coefficients"] = coefficients
        loss = float(node["energy"].get("mean_fractional_loss", 0.0))
        for z_in in (0.2, 0.5, 0.8):
            exact_q = energy_quantiles(node["energy"], z_in, loss, probability)
            predicted_q = energy_quantiles(
                predicted["energy"], z_in, loss, probability)
            tangent_q = tangent_quantiles(
                baseline["energy"], predicted["energy"], z_in, loss, probability)
            base_q = energy_quantiles(
                baseline["energy"], z_in, loss, probability)
            baseline_energy.append(float(np.mean(np.abs(base_q - exact_q))))
            corrected_energy.append(float(np.mean(np.abs(tangent_q - exact_q))))
            exact_energy.append(float(np.mean(np.abs(predicted_q - exact_q))))
            tangent_error.append(float(np.mean(np.abs(tangent_q - predicted_q))))
        exact_a = angular_quantiles(np.array([
            node["angular"]["eta1"], node["angular"]["eta2"]]), probability)
        base_a = angular_quantiles(np.array([
            baseline["angular"]["eta1"], baseline["angular"]["eta2"]]), probability)
        predicted_a = angular_quantiles(np.array([
            predicted["angular"]["eta1"], predicted["angular"]["eta2"]]), probability)
        baseline_angle.append(float(np.mean(np.abs(base_a - exact_a))))
        corrected_angle.append(float(np.mean(np.abs(predicted_a - exact_a))))

    metrics = {
        "baseline_energy_w1_p95": _percentile(baseline_energy, 0.95),
        "corrected_energy_w1_p95": _percentile(corrected_energy, 0.95),
        "exact_response_energy_w1_p95": _percentile(exact_energy, 0.95),
        "tangent_to_exact_w1_p95": _percentile(tangent_error, 0.95),
        "tangent_to_exact_w1_maximum": float(np.max(tangent_error)),
        "baseline_angular_w1_p95": _percentile(baseline_angle, 0.95),
        "corrected_angular_w1_p95": _percentile(corrected_angle, 0.95),
    }
    passed = bool(
        fitted.get("linearity_pass", False)
        and metrics["tangent_to_exact_w1_maximum"] <= TANGENT_MAXIMUM_TOLERANCE
        and metrics["corrected_energy_w1_p95"] < metrics["baseline_energy_w1_p95"]
        and metrics["exact_response_energy_w1_p95"] < metrics["baseline_energy_w1_p95"]
        and metrics["corrected_angular_w1_p95"] < metrics["baseline_angular_w1_p95"])
    return {"pass": passed, "amplitude": amplitude,
            "n_observations": len(heldout), "n_expected": expected,
            "tangent_maximum_tolerance": TANGENT_MAXIMUM_TOLERANCE,
            **metrics}
