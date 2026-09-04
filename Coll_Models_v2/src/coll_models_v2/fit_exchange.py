"""Weighted identification of the energy-partition kernel.

The kernel is the I-projection of the memoryless Borgnakke-Larsen draw onto the
measured transfer moments:

    p(z' | z, eps)  proportional to  Beta(2,2)(z')
                    * exp(lambda1 z' + lambda2 z'^2 + lambda3 z z' + lambda4 eps z')

Relative to the gated form it previously used, this adds the memory statistic
``z z'`` and the fractional-loss statistic ``eps z'`` to the sufficient
statistics.  Three consequences matter:

* there is no atom at ``z' = z``.  The stored events do not have one -- only 3
  to 13 percent of collisions leave the partition unchanged to 0.01, where the
  Bernoulli form needs 25 to 95 percent -- and forcing one imposes a floor on
  the conditional variance that the data fall below;
* the dual is strictly convex on sample moments, so the "infeasible reset
  moments" branch cannot occur;
* the rotational collision number survives as the derived lag-one slope rather
  than as a parameter the reset law has to be divided by.

``reset_mean`` and ``reset_second_moment`` are retained under their old names
for artifact compatibility, but they now report the first two moments of the
kernel's *invariant* law -- the partition the kernel drives towards, which is
what the reset law was a proxy for.  On synthetic gated-Beta data the two agree
to 0.001 at an exchange probability of 0.4 and to 0.03 at 0.2: the gate is not
a member of this family, and the weaker the exchange the fewer collisions carry
information about the law being approached.  Where that matters -- the
weakly-coupled low aspect ratios -- the fix is to impose equilibrium through a
Sinkhorn-normalised bridge rather than to infer it.
"""

from __future__ import annotations

import numpy as np

from .projections import (
    PROJECTION_TOLERANCE,
    conditional_energy_logpdf,
    conditional_energy_stationary,
    fit_conditional_energy_projection,
)


MODEL_FORM_TOLERANCE_NATS = 0.02


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * value) / np.sum(weight))


def _affine_memory(z_in: np.ndarray, z_out: np.ndarray,
                   weight: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((np.ones(len(z_in)), z_in))
    lhs = (design * weight[:, None]).T @ design
    rhs = (design * weight[:, None]).T @ z_out
    intercept, coefficient = np.linalg.solve(lhs, rhs)
    return float(intercept), float(coefficient)


def fit_exchange_kernel(z_in: np.ndarray, z_out: np.ndarray,
                        weight: np.ndarray, loss: np.ndarray | None = None,
                        quadrature: int = 256,
                        loss_spread_threshold: float = 1.0e-4,
                        model_form: bool = True,
                        initial: np.ndarray | None = None) -> dict:
    z_in, z_out, weight = map(lambda x: np.asarray(x, dtype=float),
                              (z_in, z_out, weight))
    if not (z_in.shape == z_out.shape == weight.shape) or z_in.ndim != 1:
        raise ValueError("z_in, z_out, and weight must be equal-length vectors")
    if np.any(weight <= 0.0) or np.any((z_in <= 0.0) | (z_in >= 1.0)) \
            or np.any((z_out <= 0.0) | (z_out >= 1.0)):
        raise ValueError("weights must be positive and partitions must lie in (0,1)")

    # Affine memory diagnostic. This is unchanged and is what the previous
    # p_exch was, but it is now reported rather than inverted for a reset law.
    # It is never clipped, and a value outside (0,1] no longer blocks the fit:
    # the projection is well posed for any conditional law on (0,1), so a
    # diagnostic the kernel does not use must not be able to veto it.  The QA
    # gate carries the flag instead.
    intercept, coefficient = _affine_memory(z_in, z_out, weight)
    p_exch = 1.0 - coefficient
    memory_diagnostic_pass = bool(0.0 < p_exch <= 1.0)

    covariates = [z_in]
    names = ["z_in"]
    loss_deployed = False
    if loss is not None:
        loss = np.asarray(loss, dtype=float)
        if loss.shape != z_in.shape:
            raise ValueError("loss must have one entry per event")
        mean_loss = _weighted_mean(loss, weight)
        spread = np.sqrt(max(_weighted_mean((loss - mean_loss) ** 2, weight), 0.0))
        loss_deployed = bool(spread > loss_spread_threshold)
        if loss_deployed:
            covariates.append(loss)
            names.append("loss")
    else:
        mean_loss = 0.0

    design = np.column_stack(covariates)
    projection = fit_conditional_energy_projection(
        z_out, design, weight, tuple(names), quadrature=quadrature, initial=initial)
    if projection.residual > PROJECTION_TOLERANCE:
        raise ValueError(f"energy I-projection residual {projection.residual:.3e}")
    parameters = projection.parameters
    lambda3 = float(parameters[2])
    lambda4 = float(parameters[3]) if loss_deployed else 0.0

    # Invariant law of the fitted kernel at the mean fractional loss.
    nodes, mass = conditional_energy_stationary(
        np.array([parameters[0], parameters[1], lambda3]),
        offset=lambda4 * mean_loss, quadrature=quadrature)
    stationary_mean = float(mass @ nodes)
    stationary_second = float(mass @ (nodes * nodes))

    # Held-out model form. The enriched family adds the quadratic memory
    # statistic z^2 z'; a large gain would mean the mean map is curved in a way
    # the deployed statistics cannot carry.  Bootstrap replicates skip it.
    base_score = rich_score = float("nan")
    gain = 0.0
    if model_form:
        train = np.arange(len(z_in)) % 5 != 0
        test = ~train
        enriched = np.column_stack(covariates + [z_in * z_in])
        base_fit = fit_conditional_energy_projection(
            z_out[train], design[train], weight[train], quadrature=quadrature,
            initial=parameters)
        rich_fit = fit_conditional_energy_projection(
            z_out[train], enriched[train], weight[train], quadrature=quadrature,
            initial=np.append(parameters, 0.0))
        base_score = _weighted_mean(conditional_energy_logpdf(
            base_fit.parameters, design[test], z_out[test], quadrature), weight[test])
        rich_score = _weighted_mean(conditional_energy_logpdf(
            rich_fit.parameters, enriched[test], z_out[test], quadrature), weight[test])
        gain = float(rich_score - base_score)

    return {
        "p_exch": float(p_exch),
        "memory_diagnostic_pass": memory_diagnostic_pass,
        "affine_intercept": float(intercept),
        "affine_slope": float(coefficient - 1.0),
        "lambda1": float(parameters[0]),
        "lambda2": float(parameters[1]),
        "lambda3": lambda3,
        "lambda4": lambda4,
        "loss_covariate_deployed": loss_deployed,
        "mean_fractional_loss": float(mean_loss),
        "mean_partition_out": _weighted_mean(z_out, weight),
        "stationary_mean": stationary_mean,
        "stationary_second_moment": stationary_second,
        # Compatibility aliases; see the module docstring.
        "reset_mean": stationary_mean,
        "reset_second_moment": stationary_second,
        "projection_residual": float(projection.residual),
        "projection_converged": bool(projection.converged),
        "heldout_base_log_density": float(base_score),
        "heldout_enriched_log_density": float(rich_score),
        "nonlinear_improvement": gain,
        "model_form_pass": bool(not model_form or gain < MODEL_FORM_TOLERANCE_NATS),
        "kernel_form": "conditional_iprojection_v2",
    }
