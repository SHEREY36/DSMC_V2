"""Identifiability-gated natural-parameter correction fit."""

from __future__ import annotations

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES

from .response import fit_response


def fit_lambda1_coefficients(nodes: list[dict]) -> dict:
    """Fit lambda1=lambda1,0+beta dot X at one coarse grid node."""
    baseline = [node for node in nodes if int(node.get("ensemble_id", 0)) == 0]
    if len(baseline) != 1:
        raise ValueError("coefficient fit requires exactly one baseline ensemble")
    baseline = baseline[0]
    excited = [node for node in nodes if int(node.get("ensemble_id", 0)) != 0]
    if len(excited) < len(FEATURE_NAMES):
        raise ValueError("fewer excitation ensembles than production coefficients")
    fitted = fit_response(baseline, excited, FEATURE_NAMES, "energy", "lambda1")
    beta = np.array([fitted["coefficients"][name] for name in FEATURE_NAMES])
    beta_se = np.array([fitted["coefficient_standard_errors"][name]
                        for name in FEATURE_NAMES])
    deployed = np.array([fitted["coefficient_deployed"][name]
                         for name in FEATURE_NAMES], dtype=bool)
    rank = fitted["design_rank"]
    return {
        "feature_order": list(FEATURE_NAMES),
        "feature_center": fitted["feature_center"],
        "lambda1_baseline": float(baseline["energy"]["lambda1"]),
        "beta": beta.tolist(),
        "beta_se": beta_se.tolist(),
        "beta_deployed": deployed.tolist(),
        "fit_method": "shared_baseline_gls_central_amplitudes_v1",
        "design_rank": rank,
        "condition_number": fitted["condition_number_scaled"],
        "identifiable": bool(rank == len(FEATURE_NAMES)),
        "maximum_contribution_halfwidth": [
            fitted["maximum_contribution_halfwidth"][name] for name in FEATURE_NAMES],
        "training_relative_rmse": fitted["training_relative_rmse"],
        "validation_relative_rmse": fitted["validation_relative_rmse"],
        "linearity_pass": bool(fitted["validation_relative_rmse"] is not None
                               and fitted["validation_relative_rmse"] <= 0.15),
    }
