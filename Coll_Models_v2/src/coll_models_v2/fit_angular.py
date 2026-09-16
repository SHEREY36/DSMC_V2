"""Weighted exact angular I-projection and coupling diagnostics."""

from __future__ import annotations

import numpy as np

from .projections import (
    PROJECTION_TOLERANCE,
    conditional_angular_logpdf,
    fit_angular_projection,
    fit_conditional_angular_projection,
)


def _weighted_mean(value, weight):
    return float(np.sum(weight * value) / np.sum(weight))


def _weighted_correlation(x, y, weight):
    mx, my = _weighted_mean(x, weight), _weighted_mean(y, weight)
    covariance = _weighted_mean((x - mx) * (y - my), weight)
    variance = _weighted_mean((x - mx) ** 2, weight) * _weighted_mean((y - my) ** 2, weight)
    return float(covariance / np.sqrt(max(variance, 1.0e-30)))


def fit_angular_kernel(cosine: np.ndarray, z: np.ndarray,
                       weight: np.ndarray, allow_joint: bool = True) -> dict:
    cosine, z, weight = map(lambda x: np.asarray(x, dtype=float), (cosine, z, weight))
    if not (cosine.shape == z.shape == weight.shape) or cosine.ndim != 1:
        raise ValueError("cosine, z, and weight must be equal-length vectors")
    if np.any(weight <= 0.0) or np.any(np.abs(cosine) > 1.0 + 1.0e-12):
        raise ValueError("invalid angular observations")
    p2 = 0.5 * (3.0 * cosine * cosine - 1.0)
    mean_c, mean_p2 = _weighted_mean(cosine, weight), _weighted_mean(p2, weight)
    projection = fit_angular_projection(mean_c, mean_p2)
    if projection.residual > PROJECTION_TOLERANCE:
        raise ValueError(f"angular I-projection residual {projection.residual:.3e}")
    rho = _weighted_correlation(z, cosine, weight)
    ess = np.sum(weight) ** 2 / np.sum(weight * weight)
    fisher_se = 1.0 / np.sqrt(max(ess - 3.0, 1.0))
    fisher = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    rho_ci = np.tanh([fisher - 1.96 * fisher_se, fisher + 1.96 * fisher_se])
    # Deterministic held-out split evaluates the actual normalized conditional
    # density, not an unnormalised natural-parameter score.
    train = np.arange(len(z)) % 5 != 0
    test = ~train
    train_projection = fit_angular_projection(
        _weighted_mean(cosine[train], weight[train]),
        _weighted_mean(p2[train], weight[train]))
    independent_parameter = np.r_[train_projection.parameters, 0.0]
    independent_log_score = _weighted_mean(
        conditional_angular_logpdf(cosine[test], z[test], independent_parameter),
        weight[test])
    joint = None
    improvement = 0.0
    interval_excludes_zero = bool(rho_ci[0] > 0.0 or rho_ci[1] < 0.0)
    if allow_joint and abs(rho) > 0.1 and interval_excludes_zero:
        candidate = fit_conditional_angular_projection(
            z[train], cosine[train], weight[train])
        if candidate.converged:
            joint_score = _weighted_mean(conditional_angular_logpdf(
                cosine[test], z[test], candidate.parameters), weight[test])
            improvement = (joint_score - independent_log_score) \
                / max(abs(independent_log_score), 1.0e-12)
            joint = candidate
    deploy_joint = bool(joint is not None and improvement >= 0.01)
    return {
        "mean_cosine": mean_c,
        "mean_p2": mean_p2,
        "eta1": float(projection.parameters[0]),
        "eta2": float(projection.parameters[1]),
        "projection_residual": projection.residual,
        "rho_z_cosine": rho,
        "rho_ci_low": float(rho_ci[0]),
        "rho_ci_high": float(rho_ci[1]),
        "joint_log_score_improvement": float(improvement),
        "joint_deployed": deploy_joint,
        "joint_parameters": None if not deploy_joint else joint.parameters.tolist(),
    }
