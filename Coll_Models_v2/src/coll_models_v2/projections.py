"""Exact finite-dimensional I-projections used by the closure."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize


# The release gate accepts a projection residual below 1e-6. Estimators raise
# against that same number so a solve the gate would accept is never discarded:
# a residual of 1.4e-8 matches the target moments to eight significant figures,
# and rejecting it fails every gate on the node, not just the angular kernel.
PROJECTION_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class Projection:
    parameters: np.ndarray
    moments: np.ndarray
    residual: float
    converged: bool


def _solve(stats: np.ndarray, base_weight: np.ndarray, target: np.ndarray,
           initial: np.ndarray | None = None, tolerance: float = 1.0e-12,
           max_iterations: int = 200) -> Projection:
    """I-projection of ``base_weight`` onto prescribed moments of ``stats``.

    Solved by damped Newton on the strictly convex dual. The Hessian of an
    exponential family is the covariance of its sufficient statistics under the
    tilted law, so it is available analytically and Newton converges
    quadratically to machine precision.

    A quasi-Newton search stalls here: it left 9 per cent of feasible angular
    targets above the 1e-8 convergence threshold, with residuals reaching 0.22
    near the boundary of the feasible region, and a stalled solve discards the
    whole node rather than just the angular kernel.
    """
    stats = np.asarray(stats, dtype=float)
    base_weight = np.asarray(base_weight, dtype=float)
    target = np.asarray(target, dtype=float)
    count = stats.shape[1]
    parameter = np.zeros(count) if initial is None else np.asarray(initial, dtype=float)
    log_base = np.log(base_weight)

    def distribution(value):
        logw = log_base + stats @ value
        logw -= np.max(logw)
        probability = np.exp(logw)
        return probability / np.sum(probability)

    def objective(value):
        logw = log_base + stats @ value
        maximum = np.max(logw)
        return float(maximum + np.log(np.sum(np.exp(logw - maximum))) - value @ target)

    def gradient_and_hessian(value):
        probability = distribution(value)
        moments = probability @ stats
        curvature = (stats * probability[:, None]).T @ stats - np.outer(moments, moments)
        return moments - target, curvature

    current = objective(parameter)
    for _ in range(max_iterations):
        grad, curvature = gradient_and_hessian(parameter)
        if np.max(np.abs(grad)) < tolerance:
            break
        ridge = 1.0e-12 * max(1.0, float(np.max(np.abs(np.diag(curvature)))))
        try:
            step = np.linalg.solve(curvature + ridge * np.eye(count), grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(curvature, grad, rcond=None)[0]
        scaling = 1.0
        for _ in range(60):
            candidate = parameter - scaling * step
            trial = objective(candidate)
            if np.isfinite(trial) and trial <= current - 1.0e-4 * scaling * float(grad @ step):
                break
            scaling *= 0.5
        else:
            break
        parameter, current = candidate, trial

    probability = distribution(parameter)
    moments = probability @ stats
    residual = float(np.max(np.abs(moments - target)))
    return Projection(np.asarray(parameter), np.asarray(moments), residual,
                      bool(residual < 1.0e-8 and np.all(np.isfinite(parameter))))


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


def angular_moments_are_feasible(mean_cosine: float, mean_p2: float) -> bool:
    """Can any law on [-1,1] have these first two Legendre moments?

    With w = (1 + cos chi)/2 and m = E[w], the moments are Lam1 = 2m - 1 and
    Lam2 = 6 E[w^2] - 6m + 1. Since w lies in [0,1], Jensen gives
    E[w^2] >= m^2 and boundedness gives E[w^2] <= m, so
    6m^2 - 6m + 1 <= Lam2 <= 1.
    """
    if not -1.0 < float(mean_cosine) < 1.0:
        return False
    m = 0.5 * (1.0 + float(mean_cosine))
    return bool(6.0 * m * m - 6.0 * m + 1.0 < float(mean_p2) < 1.0)


def fit_angular_projection(mean_cosine: float, mean_p2: float,
                           quadrature: int = 256) -> Projection:
    """Fit exp(eta1*c+eta2*P2(c)) relative to isotropic scattering."""
    if not angular_moments_are_feasible(mean_cosine, mean_p2):
        m = 0.5 * (1.0 + float(mean_cosine))
        raise ValueError(
            f"angular moments (Lambda1={mean_cosine:.6g}, Lambda2={mean_p2:.6g}) lie "
            f"outside the moment cone of any law on [-1,1]: Lambda2 must lie in "
            f"({6.0 * m * m - 6.0 * m + 1.0:.6g}, 1)")
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


@dataclass(frozen=True)
class ConditionalEnergyKernel:
    """Beta(2,2) tilted by z, z**2 and per-event covariates multiplying z.

    ``parameters`` is ``(lambda1, lambda2, lambda_k...)`` where the trailing
    entries multiply ``covariate_k * z``.  The first covariate is always the
    incoming partition, so ``lambda3`` is the memory parameter that replaces
    the Bernoulli gate.
    """

    parameters: np.ndarray
    moments: np.ndarray
    targets: np.ndarray
    residual: float
    converged: bool
    covariate_names: tuple[str, ...]
    quadrature: int


CONDITIONAL_ENERGY_CHUNK = 8192


def _conditional_energy_chunk(parameters, shift, z, log_base, order):
    """Log-normaliser and moments of z for one block of events."""
    exponent = (log_base + parameters[0] * z + parameters[1] * z * z)[None, :] \
        + shift[:, None] * z[None, :]
    maximum = np.max(exponent, axis=1)
    scaled = np.exp(exponent - maximum[:, None])
    total = np.sum(scaled, axis=1)
    probability = scaled / total[:, None]
    return maximum + np.log(total), [probability @ (z ** power)
                                     for power in range(1, order + 1)]


def _conditional_energy_terms(parameters: np.ndarray, covariates: np.ndarray,
                              quadrature: int, order: int = 2,
                              chunk: int = CONDITIONAL_ENERGY_CHUNK):
    """Per-event log-normaliser and the first ``order`` moments of z.

    The kernel depends on an event only through the scalar
    ``a = sum_k lambda_{k+2} * covariate_k``, so the exponent is built once per
    block.  Blocking keeps the working set at ``chunk x quadrature`` rather than
    ``n x quadrature``, which for a production node is the difference between a
    few megabytes and several gigabytes.
    """
    z, weight = _legendre_nodes(quadrature, 0.0, 1.0)
    log_base = np.log(weight * 6.0 * z * (1.0 - z))
    shift = np.asarray(covariates, dtype=float) @ np.asarray(parameters, dtype=float)[2:]
    count = len(shift)
    log_norm = np.empty(count)
    moments = [np.empty(count) for _ in range(order)]
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        block_norm, block_moments = _conditional_energy_chunk(
            parameters, shift[start:stop], z, log_base, order)
        log_norm[start:stop] = block_norm
        for target, value in zip(moments, block_moments):
            target[start:stop] = value
    return log_norm, moments, z


def conditional_energy_logpdf(parameters: np.ndarray, covariates: np.ndarray,
                              z_out: np.ndarray, quadrature: int = 256) -> np.ndarray:
    """log p(z_out | covariates) for the tilted conditional kernel."""
    parameters = np.asarray(parameters, dtype=float)
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    z_out = np.asarray(z_out, dtype=float)
    if covariates.shape[0] != z_out.shape[0]:
        covariates = covariates.T
    log_norm, _, _ = _conditional_energy_terms(parameters, covariates, quadrature, order=1)
    shift = covariates @ parameters[2:]
    return (np.log(6.0 * z_out * (1.0 - z_out)) + parameters[0] * z_out
            + parameters[1] * z_out * z_out + shift * z_out - log_norm)


def _gaussian_warm_start(z_out: np.ndarray, covariates: np.ndarray,
                         weight: np.ndarray) -> np.ndarray:
    """Closed-form starting point from the affine conditional mean and spread.

    Ignoring the Beta(2,2) factor, the tilt is Gaussian in z with variance
    ``-1/(2 lambda2)`` and mean ``-(lambda1 + sum_k lambda_{k+2} c_k)/(2 lambda2)``.
    Matching that to the weighted affine regression of z_out on the covariates
    lands within a couple of Newton steps of the solution even when the
    conditional law is very sharp, where a cold start needs hundreds of damped
    steps and can exhaust the line search.
    """
    design = np.column_stack((np.ones(len(z_out)), covariates))
    normal = (design * weight[:, None]).T @ design
    coefficients = np.linalg.solve(
        normal + 1.0e-12 * np.eye(design.shape[1]),
        (design * weight[:, None]).T @ z_out)
    residual = z_out - design @ coefficients
    variance = float(weight @ (residual * residual) / np.sum(weight))
    variance = min(max(variance, 1.0e-6), 1.0)
    lambda2 = -0.5 / variance
    return np.concatenate(([coefficients[0] / variance, lambda2],
                           coefficients[1:] / variance))


def fit_conditional_energy_projection(z_out: np.ndarray, covariates: np.ndarray,
                                      weight: np.ndarray,
                                      covariate_names: tuple[str, ...] = (),
                                      quadrature: int = 256,
                                      tolerance: float = 1.0e-11,
                                      max_iterations: int = 80,
                                      initial: np.ndarray | None = None) -> ConditionalEnergyKernel:
    """I-projection of Beta(2,2) onto E[z], E[z**2] and E[covariate_k * z].

    The dual is strictly convex, so unlike the gated-Beta inversion this has a
    unique solution whenever the targets are moments of an actual distribution
    on (0,1) -- which sample moments always are.  Solved by damped Newton with
    the exact Hessian, which is the model covariance of the statistics.
    """
    z_out = np.asarray(z_out, dtype=float)
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    if covariates.shape[0] != z_out.shape[0]:
        covariates = covariates.T
    if covariates.shape[0] != z_out.shape[0]:
        raise ValueError("covariates must have one row per event")
    weight = np.asarray(weight, dtype=float)
    weight = weight / np.sum(weight)
    count = 2 + covariates.shape[1]

    statistics = np.column_stack(
        (z_out, z_out * z_out, covariates * z_out[:, None]))
    target = weight @ statistics

    nodes, quad_weight = _legendre_nodes(quadrature, 0.0, 1.0)
    log_base = np.log(quad_weight * 6.0 * nodes * (1.0 - nodes))
    chunk = CONDITIONAL_ENERGY_CHUNK

    def dual(parameter):
        shift = covariates @ parameter[2:]
        total = 0.0
        for start in range(0, len(z_out), chunk):
            stop = min(start + chunk, len(z_out))
            block_norm, _ = _conditional_energy_chunk(
                parameter, shift[start:stop], nodes, log_base, 1)
            total += float(weight[start:stop] @ block_norm)
        return total - float(parameter @ target)

    def gradient_and_hessian(parameter):
        shift = covariates @ parameter[2:]
        model = np.zeros(count)
        variance = np.zeros((count, count))
        for start in range(0, len(z_out), chunk):
            stop = min(start + chunk, len(z_out))
            _, moments = _conditional_energy_chunk(
                parameter, shift[start:stop], nodes, log_base, 4)
            m1, m2, m3, m4 = moments
            block_weight = weight[start:stop]
            block_covariates = covariates[start:stop]
            model[0] += block_weight @ m1
            model[1] += block_weight @ m2
            for k in range(covariates.shape[1]):
                model[2 + k] += block_weight @ (block_covariates[:, k] * m1)
            v11, v12, v22 = m2 - m1 * m1, m3 - m1 * m2, m4 - m2 * m2
            # basis is (1, 1) for the two z-powers then one column per covariate
            scale = np.column_stack((np.ones_like(m1), np.ones_like(m1),
                                     block_covariates))
            for i in range(count):
                for j in range(i, count):
                    block = v22 if (i == 1 and j == 1) else (
                        v12 if (i == 1) != (j == 1) else v11)
                    value = block_weight @ (scale[:, i] * scale[:, j] * block)
                    variance[i, j] += value
                    if i != j:
                        variance[j, i] += value
        return model - target, variance

    if initial is None:
        initial = _gaussian_warm_start(z_out, covariates, weight)
    parameter = np.asarray(initial, dtype=float)
    if parameter.shape != (count,):
        parameter = np.zeros(count)
    value = dual(parameter)
    if not np.isfinite(value):
        parameter = np.zeros(count)
        value = dual(parameter)
    residual = np.inf
    budget = 12 * max_iterations
    for _ in range(max_iterations):
        grad, hessian = gradient_and_hessian(parameter)
        residual = float(np.max(np.abs(grad)))
        if residual < tolerance:
            break
        ridge = 1.0e-12 * max(1.0, float(np.max(np.abs(np.diag(hessian)))))
        step = np.linalg.solve(hessian + ridge * np.eye(count), grad)
        scaling = 1.0
        for _ in range(30):
            budget -= 1
            candidate = parameter - scaling * step
            trial = dual(candidate)
            if np.isfinite(trial) and trial <= value - 1.0e-4 * scaling * float(grad @ step):
                break
            scaling *= 0.5
        else:
            break
        parameter, value = candidate, trial
        if budget <= 0:
            break
    grad, _ = gradient_and_hessian(parameter)
    residual = float(np.max(np.abs(grad)))
    return ConditionalEnergyKernel(
        parameters=parameter, moments=target + grad, targets=target,
        residual=residual,
        converged=bool(residual < 1.0e-8 and np.all(np.isfinite(parameter))),
        covariate_names=tuple(covariate_names), quadrature=int(quadrature))


def conditional_energy_mean_map(parameters: np.ndarray, z_in: np.ndarray,
                                offset: float = 0.0, quadrature: int = 256) -> np.ndarray:
    """E[z_out | z_in] with the non-memory covariates folded into ``offset``."""
    parameters = np.asarray(parameters, dtype=float)
    z_in = np.atleast_1d(np.asarray(z_in, dtype=float))
    effective = np.array([parameters[0] + offset, parameters[1], parameters[2]])
    _, moments, _ = _conditional_energy_terms(effective, z_in[:, None], quadrature, 1)
    return moments[0]


def conditional_energy_stationary(parameters: np.ndarray, offset: float = 0.0,
                                  quadrature: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Invariant law of the z-chain, returned as (nodes, probability mass).

    This is the object the discarded "reset law" was a proxy for: the partition
    the kernel drives towards.  It is Beta(2,2) exactly for an elastic kernel
    that respects equipartition, so its first two moments are the elastic gate.
    """
    parameters = np.asarray(parameters, dtype=float)
    z, weight = _legendre_nodes(quadrature, 0.0, 1.0)
    log_base = np.log(weight * 6.0 * z * (1.0 - z))
    exponent = (log_base + (parameters[0] + offset) * z + parameters[1] * z * z)[None, :] \
        + parameters[2] * np.outer(z, z)
    scaled = np.exp(exponent - np.max(exponent, axis=1)[:, None])
    transition = scaled / np.sum(scaled, axis=1)[:, None]
    system = np.vstack((transition.T - np.eye(len(z)), np.ones(len(z))))
    rhs = np.zeros(len(z) + 1)
    rhs[-1] = 1.0
    mass, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    mass = np.maximum(mass, 0.0)
    return z, mass / np.sum(mass)


def incoming_partition_density(theta: float, z: np.ndarray) -> np.ndarray:
    """Collision-weighted law of z_in for two independent Gamma(2) modal energies.

    For X ~ Gamma(2, theta) and Y ~ Gamma(2, 1) the ratio X/(X+Y) has density
    proportional to z(1-z)(z/theta + 1 - z)^-4.
    """
    z = np.asarray(z, dtype=float)
    density = z * (1.0 - z) / (z / float(theta) + 1.0 - z) ** 4
    return density


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
