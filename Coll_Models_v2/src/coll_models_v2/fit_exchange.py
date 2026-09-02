"""Weighted identification of the gated energy-partition kernel."""

from __future__ import annotations

import numpy as np

from .projections import fit_energy_projection


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * value) / np.sum(weight))


def fit_exchange_kernel(z_in: np.ndarray, z_out: np.ndarray,
                        weight: np.ndarray) -> dict:
    z_in, z_out, weight = map(lambda x: np.asarray(x, dtype=float),
                              (z_in, z_out, weight))
    if not (z_in.shape == z_out.shape == weight.shape) or z_in.ndim != 1:
        raise ValueError("z_in, z_out, and weight must be equal-length vectors")
    if np.any(weight <= 0.0) or np.any((z_in <= 0.0) | (z_in >= 1.0)) \
            or np.any((z_out <= 0.0) | (z_out >= 1.0)):
        raise ValueError("weights must be positive and partitions must lie in (0,1)")
    design = np.column_stack((np.ones(len(z_in)), z_in))
    lhs = (design * weight[:, None]).T @ design
    rhs = (design * weight[:, None]).T @ z_out
    intercept, coefficient = np.linalg.solve(lhs, rhs)
    p_exch = 1.0 - float(coefficient)
    if not (0.0 < p_exch <= 1.0):
        raise ValueError(f"raw exchange probability is outside (0,1]: {p_exch}")
    reset_mean = float(intercept / p_exch)
    zin2 = _weighted_mean(z_in * z_in, weight)
    zout2 = _weighted_mean(z_out * z_out, weight)
    reset_second = (zout2 - (1.0 - p_exch) * zin2) / p_exch
    projection = fit_energy_projection(reset_mean, reset_second)
    if not projection.converged:
        raise ValueError(f"energy I-projection residual {projection.residual:.3e}")

    # Held-out flexible check. A fifth-order Legendre regression is expressive
    # on the compact z interval while remaining deterministic and inexpensive.
    train = np.arange(len(z_in)) % 5 != 0
    test = ~train
    affine_design = np.column_stack((np.ones(np.sum(train)), z_in[train]))
    affine_coef = np.linalg.solve(
        (affine_design * weight[train, None]).T @ affine_design,
        (affine_design * weight[train, None]).T @ z_out[train])
    flexible_design = np.polynomial.legendre.legvander(2.0 * z_in[train] - 1.0, 5)
    flexible_lhs = (flexible_design * weight[train, None]).T @ flexible_design
    flexible_lhs += 1.0e-6 * np.diag([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    flexible_coef = np.linalg.solve(
        flexible_lhs, (flexible_design * weight[train, None]).T @ z_out[train])
    affine_prediction = affine_coef[0] + affine_coef[1] * z_in[test]
    flexible_prediction = np.polynomial.legendre.legvander(
        2.0 * z_in[test] - 1.0, 5) @ flexible_coef
    affine_mse = _weighted_mean((z_out[test] - affine_prediction) ** 2, weight[test])
    flexible_mse = _weighted_mean((z_out[test] - flexible_prediction) ** 2, weight[test])
    improvement = max(0.0, (affine_mse - flexible_mse) / max(affine_mse, 1.0e-30))
    return {
        "p_exch": p_exch,
        "affine_intercept": float(intercept),
        "affine_slope": float(coefficient - 1.0),
        "reset_mean": reset_mean,
        "reset_second_moment": float(reset_second),
        "lambda1": float(projection.parameters[0]),
        "lambda2": float(projection.parameters[1]),
        "projection_residual": projection.residual,
        "projection_converged": projection.converged,
        "heldout_affine_mse": affine_mse,
        "heldout_flexible_mse": flexible_mse,
        "nonlinear_improvement": improvement,
        "model_form_pass": bool(improvement < 0.02),
    }
