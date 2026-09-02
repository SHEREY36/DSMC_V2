"""Exact finite-dimensional I-projections used by the closure."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Projection:
    parameters: np.ndarray
    moments: np.ndarray
    residual: float
    converged: bool


def _solve(stats: np.ndarray, base_weight: np.ndarray, target: np.ndarray,
           initial: np.ndarray | None = None) -> Projection:
    stats = np.asarray(stats, dtype=float)
    base_weight = np.asarray(base_weight, dtype=float)
    target = np.asarray(target, dtype=float)
    initial = np.zeros(stats.shape[1]) if initial is None else np.asarray(initial, dtype=float)

    def distribution(parameter):
        logw = np.log(base_weight) + stats @ parameter
        logw -= np.max(logw)
        probability = np.exp(logw)
        probability /= np.sum(probability)
        return probability

    def objective(parameter):
        logw = np.log(base_weight) + stats @ parameter
        maximum = np.max(logw)
        logz = maximum + np.log(np.sum(np.exp(logw - maximum)))
        return float(logz - parameter @ target)

    def jacobian(parameter):
        return distribution(parameter) @ stats - target

    result = minimize(objective, initial, jac=jacobian, method="BFGS",
                      options={"gtol": 2.0e-11, "maxiter": 2000})
    probability = distribution(result.x)
    moments = probability @ stats
    residual = float(np.max(np.abs(moments - target)))
    return Projection(np.asarray(result.x), np.asarray(moments), residual,
                      bool(residual < 1.0e-8 and np.all(np.isfinite(result.x))))


@lru_cache(maxsize=16)
def _legendre_nodes(count: int, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(count))
    x = lower + 0.5 * (nodes + 1.0) * (upper - lower)
    return x, 0.5 * (upper - lower) * weights


def fit_energy_projection(mean: float, second_moment: float,
                          quadrature: int = 256) -> Projection:
    """Fit Beta(2,2)*exp(lambda1*z+lambda2*z**2) to ordinary moments."""
    mean, second_moment = float(mean), float(second_moment)
    if not (0.0 < mean < 1.0 and mean * mean < second_moment < mean):
        raise ValueError("infeasible moments on (0,1)")
    z, weight = _legendre_nodes(quadrature, 0.0, 1.0)
    base = weight * 6.0 * z * (1.0 - z)
    return _solve(np.column_stack((z, z * z)), base,
                  np.array([mean, second_moment]))


def fit_angular_projection(mean_cosine: float, mean_p2: float,
                           quadrature: int = 256) -> Projection:
    """Fit exp(eta1*c+eta2*P2(c)) relative to isotropic scattering."""
    c, weight = _legendre_nodes(quadrature, -1.0, 1.0)
    p2 = 0.5 * (3.0 * c * c - 1.0)
    return _solve(np.column_stack((c, p2)), 0.5 * weight,
                  np.array([mean_cosine, mean_p2]))


def energy_moments(parameters: np.ndarray, quadrature: int = 256) -> np.ndarray:
    """Ordinary first and second moments of a deployed energy tilt."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (2,):
        raise ValueError("energy projection requires two natural parameters")
    z, weight = _legendre_nodes(quadrature, 0.0, 1.0)
    exponent = parameters[0] * z + parameters[1] * z * z
    raw = weight * 6.0 * z * (1.0 - z) * np.exp(exponent - np.max(exponent))
    probability = raw / np.sum(raw)
    return np.array([probability @ z, probability @ (z * z)])


def fit_joint_projection(target: np.ndarray, quadrature: int = 96) -> Projection:
    """Fit the optional z*cos(chi) coupled I-projection.

    Target order is E[z], E[z^2], E[c], E[P2(c)], E[z*c].
    """
    target = np.asarray(target, dtype=float)
    if target.shape != (5,):
        raise ValueError("joint projection needs five target moments")
    z, wz = _legendre_nodes(quadrature, 0.0, 1.0)
    c, wc = _legendre_nodes(quadrature, -1.0, 1.0)
    zz, cc = np.meshgrid(z, c, indexing="ij")
    p2 = 0.5 * (3.0 * cc * cc - 1.0)
    base = (wz[:, None] * 6.0 * zz * (1.0 - zz) * 0.5 * wc[None, :]).ravel()
    stats = np.column_stack((zz.ravel(), (zz * zz).ravel(), cc.ravel(),
                             p2.ravel(), (zz * cc).ravel()))
    return _solve(stats, base, target)


def fit_conditional_angular_projection(z: np.ndarray, cosine: np.ndarray,
                                       weight: np.ndarray,
                                       quadrature: int = 128,
                                       chunk_size: int = 4096) -> Projection:
    """Fit p(c|z) proportional to exp(eta1*c+eta2*P2(c)+xi*z*c).

    Treating the observed/reset energy marginal as fixed prevents the coupling
    term from changing the separately fitted energy I-projection.
    """
    z, cosine, weight = map(lambda value: np.asarray(value, dtype=float),
                            (z, cosine, weight))
    weight = weight / np.sum(weight)
    c, wc = _legendre_nodes(quadrature, -1.0, 1.0)
    wc = 0.5 * wc
    p2 = 0.5 * (3.0 * c * c - 1.0)
    target = np.array([np.sum(weight * cosine),
                       np.sum(weight * 0.5 * (3.0 * cosine * cosine - 1.0)),
                       np.sum(weight * z * cosine)])

    def objective(parameter):
        total = 0.0
        for start in range(0, len(z), chunk_size):
            stop = min(start + chunk_size, len(z))
            zs = z[start:stop]
            exponent = ((parameter[0] + parameter[2] * zs[:, None]) * c[None, :]
                        + parameter[1] * p2[None, :])
            maximum = np.max(exponent, axis=1)
            logz = maximum + np.log(np.sum(wc[None, :] * np.exp(
                exponent - maximum[:, None]), axis=1))
            total += float(np.sum(weight[start:stop] * logz))
        return total - float(parameter @ target)

    def jacobian(parameter):
        moments = np.zeros(3)
        for start in range(0, len(z), chunk_size):
            stop = min(start + chunk_size, len(z))
            zs, ws = z[start:stop], weight[start:stop]
            exponent = ((parameter[0] + parameter[2] * zs[:, None]) * c[None, :]
                        + parameter[1] * p2[None, :])
            maximum = np.max(exponent, axis=1)
            raw = wc[None, :] * np.exp(exponent - maximum[:, None])
            probability = raw / np.sum(raw, axis=1)[:, None]
            mean_c, mean_p2 = probability @ c, probability @ p2
            moments += [np.sum(ws * mean_c), np.sum(ws * mean_p2),
                        np.sum(ws * zs * mean_c)]
        return moments - target

    result = minimize(objective, np.zeros(3), jac=jacobian, method="BFGS",
                      options={"gtol": 2.0e-10, "maxiter": 1000})
    residual = float(np.max(np.abs(jacobian(result.x))))
    return Projection(np.asarray(result.x), target + jacobian(result.x), residual,
                      bool(residual < 1.0e-7 and np.all(np.isfinite(result.x))))


def conditional_angular_logpdf(cosine: np.ndarray, z: np.ndarray,
                               parameters: np.ndarray,
                               quadrature: int = 128,
                               chunk_size: int = 4096) -> np.ndarray:
    cosine, z = np.asarray(cosine, dtype=float), np.asarray(z, dtype=float)
    c, wc = _legendre_nodes(quadrature, -1.0, 1.0)
    wc = 0.5 * wc
    p2 = 0.5 * (3.0 * c * c - 1.0)
    logz = np.empty(len(z))
    for start in range(0, len(z), chunk_size):
        stop = min(start + chunk_size, len(z))
        exponent = ((parameters[0] + parameters[2] * z[start:stop, None]) * c[None, :]
                    + parameters[1] * p2[None, :])
        maximum = np.max(exponent, axis=1)
        logz[start:stop] = maximum + np.log(np.sum(wc[None, :] * np.exp(
            exponent - maximum[:, None]), axis=1))
    observed_p2 = 0.5 * (3.0 * cosine * cosine - 1.0)
    return (-np.log(2.0) + (parameters[0] + parameters[2] * z) * cosine
            + parameters[1] * observed_p2 - logz)


def energy_quantiles(parameters: np.ndarray, probabilities: np.ndarray,
                     grid_size: int = 16385) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    z = np.linspace(0.0, 1.0, int(grid_size))
    density = 6.0 * z * (1.0 - z) * np.exp(
        np.clip(parameters[0] * z + parameters[1] * z * z, -700.0, 700.0))
    cdf = np.zeros_like(z)
    cdf[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(z))
    cdf /= cdf[-1]
    return np.interp(probabilities, cdf, z)


def angular_quantiles(parameters: np.ndarray, probabilities: np.ndarray,
                      grid_size: int = 16385) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    c = np.linspace(-1.0, 1.0, int(grid_size))
    p2 = 0.5 * (3.0 * c * c - 1.0)
    density = np.exp(np.clip(parameters[0] * c + parameters[1] * p2, -700.0, 700.0))
    cdf = np.zeros_like(c)
    cdf[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(c))
    cdf /= cdf[-1]
    return np.interp(probabilities, cdf, c)
