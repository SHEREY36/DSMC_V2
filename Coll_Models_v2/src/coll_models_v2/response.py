"""Generalised least-squares response fits for virtual excitation ensembles.

Every excited estimate at a physical node is formed from the same baseline
sample.  Consequently ``lambda(excited) - lambda(baseline)`` observations do
not have independent errors: they share the uncertainty of the baseline.
This module keeps that covariance and reserves the large-amplitude points for
an actual linearity check rather than fitting and judging the same data.
"""

from __future__ import annotations

import numpy as np


CENTRAL_AMPLITUDE = 0.25
VALIDATION_AMPLITUDE = 0.50


def _value(node: dict, section: str, name: str) -> float:
    return float(node[section][name])


def _standard_error(node: dict, name: str) -> float:
    value = node.get("uncertainty", {}).get(name, {}).get("standard_error", np.nan)
    return float(value) if value is not None else np.nan


def fit_response(baseline: dict, excited: list[dict], feature_names,
                 section: str, parameter: str) -> dict:
    """Fit central amplitudes with shared-baseline GLS and test on extremes."""
    feature_names = tuple(feature_names)
    base_features = baseline.get("cell_features") or baseline["proposal_features"]
    base_x = np.array([base_features[name] for name in feature_names], dtype=float)
    usable = [node for node in excited
              if parameter in node.get(section, {}) and node.get("excitation")]
    if not usable:
        raise ValueError(f"no excitation values for {section}.{parameter}")
    x_all = np.array([
        [(node.get("cell_features") or node["proposal_features"])[name]
         for name in feature_names] for node in usable
    ], dtype=float) - base_x
    y_all = np.array([_value(node, section, parameter)
                      - _value(baseline, section, parameter) for node in usable])
    amplitude = np.abs(np.array([node["excitation"]["eta"] for node in usable],
                                dtype=float))
    train = np.isclose(amplitude, CENTRAL_AMPLITUDE)
    validate = np.isclose(amplitude, VALIDATION_AMPLITUDE)
    if np.sum(train) < len(feature_names):
        raise ValueError("too few central-amplitude observations for response fit")

    x, y = x_all[train], y_all[train]
    excited_se = np.array([_standard_error(node, parameter) for node in usable])[train]
    base_se = _standard_error(baseline, parameter)
    finite = excited_se[np.isfinite(excited_se) & (excited_se > 0.0)]
    floor = max(1.0e-12, 0.1 * float(np.median(finite)) if len(finite) else 1.0)
    excited_se = np.maximum(np.where(np.isfinite(excited_se), excited_se, floor), floor)
    base_se = base_se if np.isfinite(base_se) and base_se > 0.0 else floor
    covariance_y = np.diag(excited_se * excited_se) \
        + float(base_se * base_se) * np.ones((len(y), len(y)))
    precision = np.linalg.pinv(covariance_y, rcond=1.0e-12)
    information = x.T @ precision @ x
    information_inverse = np.linalg.pinv(information, rcond=1.0e-12)
    beta = information_inverse @ x.T @ precision @ y
    beta_se = np.sqrt(np.maximum(np.diag(information_inverse), 0.0))

    def score(mask: np.ndarray) -> tuple[float | None, float | None]:
        if not np.any(mask):
            return None, None
        residual = y_all[mask] - x_all[mask] @ beta
        rmse = float(np.sqrt(np.mean(residual * residual)))
        response_scale = max(float(np.ptp(y_all[mask])),
                             float(np.max(np.abs(y_all[mask]))), 1.0e-12)
        return rmse, rmse / response_scale

    train_rmse, train_relative = score(train)
    validation_rmse, validation_relative = score(validate)
    max_x = np.max(np.abs(x), axis=0)
    contribution_halfwidth = 1.96 * beta_se * max_x
    effect = np.abs(beta) * max_x
    # A term is released only if it is resolved from zero and its uncertainty
    # is below 25% of its largest calibrated contribution.  The 0.005 floor
    # avoids an ill-posed relative test for a genuinely tiny contribution.
    deployed = ((np.abs(beta) > 1.96 * beta_se)
                & (contribution_halfwidth <= np.maximum(0.005, 0.25 * effect)))
    all_excited_se = np.array([_standard_error(node, parameter)
                               for node in usable])
    all_excited_se = np.maximum(
        np.where(np.isfinite(all_excited_se), all_excited_se, floor), floor)
    maximum_standardized_shift = float(np.max(
        np.abs(y_all) / np.maximum(
            np.hypot(base_se, all_excited_se), 1.0e-30)))
    return {
        "feature_order": list(feature_names),
        "feature_center": base_x.tolist(),
        "coefficients": dict(zip(feature_names, beta.tolist())),
        "coefficient_standard_errors": dict(zip(feature_names, beta_se.tolist())),
        "coefficient_deployed": dict(zip(feature_names, deployed.tolist())),
        "design_rank": int(np.linalg.matrix_rank(x)),
        "condition_number_scaled": float(np.linalg.cond(
            x / np.where(np.linalg.norm(x, axis=0) > 0.0,
                         np.linalg.norm(x, axis=0), 1.0))),
        "n_training": int(np.sum(train)),
        "n_validation": int(np.sum(validate)),
        "maximum_standardized_shift": maximum_standardized_shift,
        "material_response": maximum_standardized_shift >= 3.0,
        "training_rmse": train_rmse,
        "training_relative_rmse": train_relative,
        "validation_rmse": validation_rmse,
        "validation_relative_rmse": validation_relative,
        "maximum_contribution_halfwidth": dict(zip(
            feature_names, contribution_halfwidth.tolist())),
    }
