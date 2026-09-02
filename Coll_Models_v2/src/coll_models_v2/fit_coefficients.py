"""Identifiability-gated natural-parameter correction fit."""

from __future__ import annotations

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES


def fit_lambda1_coefficients(nodes: list[dict]) -> dict:
    """Fit lambda1=lambda1,0+beta dot X at one coarse grid node."""
    baseline = [node for node in nodes if int(node.get("ensemble_id", 0)) == 0]
    if len(baseline) != 1:
        raise ValueError("coefficient fit requires exactly one baseline ensemble")
    baseline = baseline[0]
    excited = [node for node in nodes if int(node.get("ensemble_id", 0)) != 0]
    if len(excited) < len(FEATURE_NAMES):
        raise ValueError("fewer excitation ensembles than production coefficients")
    x = np.array([[node["proposal_features"][name] for name in FEATURE_NAMES]
                  for node in excited])
    y = np.array([node["energy"]["lambda1"] - baseline["energy"]["lambda1"]
                  for node in excited])
    se = np.array([node.get("uncertainty", {}).get("lambda1", {}).get(
        "standard_error", np.nan) for node in excited], dtype=float)
    finite = se[np.isfinite(se) & (se > 0.0)]
    floor = 0.1 * np.median(finite) if len(finite) else 1.0
    se = np.maximum(np.where(np.isfinite(se), se, floor), floor)
    weight = 1.0 / (se * se)
    # Deterministic five-fold CV over a conservative ridge grid.
    ridges = np.logspace(-8, 1, 24)
    folds = np.arange(len(x)) % min(5, len(x))
    errors = []
    for ridge in ridges:
        fold_error = 0.0
        for fold in np.unique(folds):
            train, test = folds != fold, folds == fold
            lhs = (x[train] * weight[train, None]).T @ x[train] + ridge * np.eye(x.shape[1])
            beta = np.linalg.solve(lhs, (x[train] * weight[train, None]).T @ y[train])
            fold_error += float(np.sum(weight[test] * (y[test] - x[test] @ beta) ** 2))
        errors.append(fold_error)
    ridge = float(ridges[int(np.argmin(errors))])
    lhs = (x * weight[:, None]).T @ x + ridge * np.eye(x.shape[1])
    beta = np.linalg.solve(lhs, (x * weight[:, None]).T @ y)
    covariance = np.linalg.inv(lhs)
    beta_se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    max_contribution_halfwidth = 1.96 * beta_se * np.max(np.abs(x), axis=0)
    deployed = (np.abs(beta) > 1.96 * beta_se) & (max_contribution_halfwidth <= 0.005)
    rank = int(np.linalg.matrix_rank(x))
    return {
        "feature_order": list(FEATURE_NAMES),
        "lambda1_baseline": float(baseline["energy"]["lambda1"]),
        "beta": beta.tolist(),
        "beta_se": beta_se.tolist(),
        "beta_deployed": deployed.tolist(),
        "ridge": ridge,
        "design_rank": rank,
        "condition_number": float(np.linalg.cond(x)),
        "identifiable": bool(rank == len(FEATURE_NAMES)),
        "maximum_contribution_halfwidth": max_contribution_halfwidth.tolist(),
    }
