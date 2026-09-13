"""Identifiability-gated multivariate natural-parameter response fits."""

from __future__ import annotations

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES

from .response import fit_response


CORRECTION_PARAMETERS = (
    ("energy", "lambda1"),
    ("energy", "lambda2"),
    ("energy", "lambda3"),
    ("energy", "lambda4"),
    ("angular", "eta1"),
    ("angular", "eta2"),
)
CORRECTION_PARAMETER_NAMES = tuple(name for _, name in CORRECTION_PARAMETERS)
LINEARITY_TOLERANCE = 0.15


def fit_correction_coefficients(nodes: list[dict]) -> dict:
    """Fit all six natural-parameter responses at one physical grid node.

    Central ``|eta|=0.25`` excitations determine the Jacobian and the
    ``|eta|=0.5`` points are held out. At the elastic plane, detailed balance
    fixes lambda1, lambda2, and lambda4 exactly; their response rows are set to
    zero instead of letting numerical noise violate that structural limit.
    """
    baseline_rows = [node for node in nodes if int(node.get("ensemble_id", 0)) == 0]
    if len(baseline_rows) != 1:
        raise ValueError("coefficient fit requires exactly one baseline ensemble")
    baseline = baseline_rows[0]
    excited = [node for node in nodes if int(node.get("ensemble_id", 0)) != 0]
    if len(excited) < len(FEATURE_NAMES):
        raise ValueError("fewer excitation ensembles than production coefficients")

    fits = {}
    beta, beta_se, deployed = [], [], []
    elastic = np.isclose(float(baseline["alpha"]), 1.0, atol=1.0e-12, rtol=0.0)
    structurally_zero = {"lambda1", "lambda2", "lambda4"} if elastic else set()
    for section, name in CORRECTION_PARAMETERS:
        fitted = fit_response(baseline, excited, FEATURE_NAMES, section, name)
        row = np.array([fitted["coefficients"][feature] for feature in FEATURE_NAMES])
        row_se = np.array([
            fitted["coefficient_standard_errors"][feature]
            for feature in FEATURE_NAMES
        ])
        row_deployed = np.array([
            fitted["coefficient_deployed"][feature]
            for feature in FEATURE_NAMES
        ], dtype=bool)
        if name in structurally_zero:
            row[:] = 0.0
            row_deployed[:] = False
            fitted["elastic_constraint_applied"] = True
            fitted["immaterial_response_suppressed"] = False
            fitted["linearity_pass"] = True
        elif not fitted["material_response"]:
            # A Jacobian row that cannot be resolved from zero is not a
            # correction.  Deploying all fourteen noisy coefficients merely
            # amplifies feature noise and can make an otherwise harmless
            # held-out relative error block the entire physical node.
            row[:] = 0.0
            row_deployed[:] = False
            fitted["elastic_constraint_applied"] = False
            fitted["immaterial_response_suppressed"] = True
            fitted["linearity_pass"] = True
        else:
            # Release the response as a jointly validated Jacobian. Dropping
            # individually non-significant columns after a full-rank fit
            # biases the multivariate prediction; the held-out response error,
            # not fourteen separate t-tests, is the correct model-level gate.
            row_deployed[:] = True
            fitted["elastic_constraint_applied"] = False
            fitted["immaterial_response_suppressed"] = False
            fitted["linearity_pass"] = bool(
                fitted["validation_relative_rmse"] is not None
                and fitted["validation_relative_rmse"] <= LINEARITY_TOLERANCE)
        fits[name] = fitted
        beta.append(row)
        beta_se.append(row_se)
        deployed.append(row_deployed)

    ranks = {fit["design_rank"] for fit in fits.values()}
    conditions = [fit["condition_number_scaled"] for fit in fits.values()]
    validation = {
        name: fit["validation_relative_rmse"] for name, fit in fits.items()
    }
    return {
        "feature_order": list(FEATURE_NAMES),
        "parameter_order": list(CORRECTION_PARAMETER_NAMES),
        "feature_center": next(iter(fits.values()))["feature_center"],
        "parameter_baseline": [
            float(baseline[section][name]) for section, name in CORRECTION_PARAMETERS
        ],
        "beta": np.asarray(beta).tolist(),
        "beta_se": np.asarray(beta_se).tolist(),
        "beta_deployed": np.asarray(deployed).tolist(),
        "fit_method": "shared_baseline_gls_central_amplitudes_multivariate_v2",
        "design_rank": min(ranks),
        "condition_number": max(conditions),
        "identifiable": bool(ranks == {len(FEATURE_NAMES)}),
        "parameter_fits": fits,
        "validation_relative_rmse": validation,
        "maximum_validation_relative_rmse": max(
            value for value in validation.values() if value is not None),
        "linearity_pass": bool(all(fit["linearity_pass"] for fit in fits.values())),
        "elastic_constraints": sorted(structurally_zero),
    }


def fit_lambda1_coefficients(nodes: list[dict]) -> dict:
    """Backward-compatible lambda1 view used by older callers/tests."""
    baseline_rows = [node for node in nodes if int(node.get("ensemble_id", 0)) == 0]
    if len(baseline_rows) != 1:
        raise ValueError("coefficient fit requires exactly one baseline ensemble")
    baseline = baseline_rows[0]
    excited = [node for node in nodes if int(node.get("ensemble_id", 0)) != 0]
    fitted = fit_response(baseline, excited, FEATURE_NAMES, "energy", "lambda1")
    beta = np.array([fitted["coefficients"][name] for name in FEATURE_NAMES])
    beta_se = np.array([
        fitted["coefficient_standard_errors"][name] for name in FEATURE_NAMES
    ])
    deployed = np.array([
        fitted["coefficient_deployed"][name] for name in FEATURE_NAMES
    ], dtype=bool)
    return {
        "feature_order": list(FEATURE_NAMES),
        "feature_center": fitted["feature_center"],
        "lambda1_baseline": float(baseline["energy"]["lambda1"]),
        "beta": beta.tolist(),
        "beta_se": beta_se.tolist(),
        "beta_deployed": deployed.tolist(),
        "fit_method": "shared_baseline_gls_central_amplitudes_v1",
        "design_rank": fitted["design_rank"],
        "condition_number": fitted["condition_number_scaled"],
        "identifiable": bool(fitted["design_rank"] == len(FEATURE_NAMES)),
        "maximum_contribution_halfwidth": [
            fitted["maximum_contribution_halfwidth"][name] for name in FEATURE_NAMES],
        "training_relative_rmse": fitted["training_relative_rmse"],
        "validation_relative_rmse": fitted["validation_relative_rmse"],
        "linearity_pass": bool(fitted["validation_relative_rmse"] is not None
                               and fitted["validation_relative_rmse"]
                               <= LINEARITY_TOLERANCE),
    }
