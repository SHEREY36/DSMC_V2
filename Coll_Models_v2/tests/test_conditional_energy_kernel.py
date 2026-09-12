"""Properties of the conditional I-projection energy kernel.

These cover the three things the gated-Beta inversion could not do: fit a
conditional law with no atom, fit one whose conditional variance is below the
Bernoulli floor, and report an invariant law rather than a reset law.
"""

import numpy as np
import pytest

from coll_models_v2.artifact import _incoming_partition_mean
from coll_models_v2.fit_exchange import fit_exchange_kernel
from coll_models_v2.projections import (
    _legendre_nodes,
    conditional_energy_logpdf,
    conditional_energy_mean_map,
    conditional_energy_stationary,
    fit_conditional_energy_projection,
    incoming_partition_density,
)


def test_untilted_kernel_has_beta22_as_its_invariant_law():
    nodes, mass = conditional_energy_stationary(np.zeros(3))
    assert mass @ nodes == pytest.approx(0.5, abs=1.0e-9)
    assert mass @ (nodes * nodes) == pytest.approx(0.3, abs=1.0e-9)


def _sample_conditional(parameters, z_in, rng):
    """Draw z_out from the conditional kernel by per-event inverse CDF."""
    edges = np.linspace(0.0, 1.0, 2049)
    base = 6.0 * edges * (1.0 - edges)
    draws = np.empty(len(z_in))
    for start in range(0, len(z_in), 4096):
        stop = min(start + 4096, len(z_in))
        shift = parameters[2] * z_in[start:stop]
        exponent = (np.log(np.maximum(base, 1.0e-300))
                    + parameters[0] * edges + parameters[1] * edges * edges)[None, :] \
            + shift[:, None] * edges[None, :]
        density = np.exp(exponent - exponent.max(axis=1, keepdims=True))
        cumulative = np.cumsum(0.5 * (density[:, 1:] + density[:, :-1])
                               * np.diff(edges)[None, :], axis=1)
        cumulative = np.concatenate((np.zeros((len(shift), 1)), cumulative), axis=1)
        cumulative /= cumulative[:, -1:]
        uniform = rng.random(stop - start)
        index = np.clip(np.sum(cumulative < uniform[:, None], axis=1) - 1,
                        0, len(edges) - 2)
        lower = cumulative[np.arange(len(index)), index]
        upper = cumulative[np.arange(len(index)), index + 1]
        fraction = np.where(upper > lower, (uniform - lower) / (upper - lower), 0.0)
        draws[start:stop] = edges[index] + fraction * np.diff(edges)[index]
    return np.clip(draws, 1.0e-9, 1.0 - 1.0e-9)


@pytest.mark.parametrize("truth", [
    np.array([-2.4, -0.8, 6.2]),
    np.array([1.5, -4.0, 3.0]),
])
def test_estimator_recovers_its_own_kernel(truth):
    """Self-consistency: data drawn from the kernel must return its parameters.

    The three natural parameters trade off against one another, so the residual
    is sampling noise rather than bias: over three seeds the largest component
    error is 0.41 at 20,000 events and 0.11 at 80,000, centred on zero.
    """
    rng = np.random.default_rng(21)
    z_in = rng.beta(2.0, 2.0, 80000)
    z_out = _sample_conditional(truth, z_in, rng)
    fit = fit_conditional_energy_projection(z_out, z_in[:, None], np.ones(len(z_in)))
    assert fit.converged
    np.testing.assert_allclose(fit.parameters, truth, atol=0.2)


@pytest.mark.parametrize("lambda3", [1.0, 4.0, 12.0])
def test_fitted_kernel_reproduces_the_observed_lag_one_slope(lambda3):
    """E[z_in z_out] is a matched moment, so the memory slope is matched exactly.

    Note that lambda3 alone is *not* a clean memory dial: with lambda1 and
    lambda2 held at zero the tilt also drags the marginal towards z=1, and the
    mean-map slope is non-monotone in lambda3. Memory is only cleanly separated
    once the marginal is pinned, which is what the Sinkhorn-normalised bridge
    form does. What the estimator guarantees is this matching property.
    """
    rng = np.random.default_rng(22)
    z_in = rng.beta(2.0, 2.0, 60000)
    z_out = _sample_conditional(np.array([-1.0, -1.0, lambda3]), z_in, rng)
    fit = fit_conditional_energy_projection(z_out, z_in[:, None], np.ones(len(z_in)))
    grid, _ = _legendre_nodes(256, 0.0, 1.0)
    mean_map = conditional_energy_mean_map(fit.parameters, z_in)
    observed = np.polyfit(z_in, z_out, 1)[0]
    modelled = np.polyfit(z_in, mean_map, 1)[0]
    assert modelled == pytest.approx(observed, abs=1.0e-6)


def test_projection_matches_its_target_moments_exactly():
    rng = np.random.default_rng(11)
    n = 40000
    z_in = rng.beta(2.0, 3.0, n)
    z_out = np.clip(0.4 * z_in + 0.3 + 0.05 * rng.standard_normal(n), 1e-6, 1 - 1e-6)
    covariates = z_in[:, None]
    fit = fit_conditional_energy_projection(z_out, covariates, np.ones(n))
    assert fit.converged
    assert fit.residual < 1.0e-9
    np.testing.assert_allclose(fit.moments, fit.targets, atol=1.0e-9)


def test_small_nudge_kernel_is_fitted_where_the_gated_form_was_infeasible():
    """Conditional variance below the Bernoulli floor: the AR=1.1 failure mode.

    A kernel that contracts the partition slightly on every collision has the
    same affine mean map as a rare full re-randomisation but far less
    conditional spread. The gated inversion has to attribute that shortfall to
    a negative reset variance; this family simply fits it.
    """
    rng = np.random.default_rng(12)
    n = 60000
    z_in = rng.beta(2.0, 2.0, n)
    z_out = np.clip(0.9 * z_in + 0.05 + 0.03 * rng.standard_normal(n), 1e-6, 1 - 1e-6)
    fit = fit_exchange_kernel(z_in, z_out, np.ones(n))
    assert fit["projection_converged"]
    assert fit["projection_residual"] < 1.0e-8
    assert fit["memory_diagnostic_pass"]
    assert 0.0 < fit["stationary_mean"] < 1.0
    assert fit["stationary_mean"] ** 2 < fit["stationary_second_moment"]
    assert fit["lambda3"] > 10.0        # strong memory, tiny per-collision move

    # The gated inversion this replaced demanded a spread the data do not have,
    # so solving it for the reset variance returns a negative number.
    intercept, slope = fit["affine_intercept"], 1.0 + fit["affine_slope"]
    p_exch = 1.0 - slope
    assert 0.0 < p_exch < 1.0
    residual = z_out - (intercept + slope * z_in)
    reset_mean = intercept / p_exch
    bernoulli_floor = p_exch * (1.0 - p_exch) * np.mean((z_in - reset_mean) ** 2)
    assert np.mean(residual ** 2) < bernoulli_floor
    implied_reset_variance = (np.mean(residual ** 2) - bernoulli_floor) / p_exch
    assert implied_reset_variance < 0.0


def test_loss_covariate_is_only_deployed_when_the_loss_varies():
    rng = np.random.default_rng(13)
    n = 20000
    z_in = rng.beta(2.0, 2.0, n)
    z_out = np.clip(0.8 * z_in + 0.1 + 0.05 * rng.standard_normal(n), 1e-6, 1 - 1e-6)
    elastic = fit_exchange_kernel(z_in, z_out, np.ones(n), loss=np.zeros(n))
    assert not elastic["loss_covariate_deployed"]
    assert elastic["lambda4"] == 0.0
    inelastic = fit_exchange_kernel(z_in, z_out, np.ones(n), loss=rng.random(n) * 0.3)
    assert inelastic["loss_covariate_deployed"]


def test_conditional_density_integrates_to_one():
    grid, quadrature = _legendre_nodes(512, 0.0, 1.0)
    for parameters in (np.array([0.0, 0.0, 0.0]),
                       np.array([-2.4, -0.8, 6.2]),
                       np.array([-7.9, -60.0, 150.0])):
        for z_in in (0.15, 0.5, 0.85):
            density = np.exp(conditional_energy_logpdf(
                parameters, np.full((len(grid), 1), z_in), grid))
            assert quadrature @ density == pytest.approx(1.0, abs=2.0e-6)


@pytest.mark.parametrize("theta", [0.1, 0.2, 1.0, 2.0, 5.0])
def test_incoming_partition_density_reproduces_its_quadrature_mean(theta):
    grid, quadrature = _legendre_nodes(256, 0.0, 1.0)
    mass = incoming_partition_density(theta, grid) * quadrature
    mass = mass / mass.sum()
    # The Laguerre reference in artifact.py is itself only good to ~1e-5 at the
    # ends of the grid, so this is a cross-check of two quadratures, not of one
    # against an exact value.
    assert mass @ grid == pytest.approx(_incoming_partition_mean(theta), abs=3.0e-5)
