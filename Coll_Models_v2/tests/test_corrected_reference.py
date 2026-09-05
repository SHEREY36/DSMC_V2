"""The 28 numerical checks from closure_reference.py under corrected semantics."""

from math import gamma

import numpy as np
import pytest
from scipy.optimize import brentq

from coll_models_v2.artifact import (
    _energy_weighted_partition,
    _incoming_partition_mean,
    _stability_rows,
)
from coll_models_v2.fit_exchange import fit_exchange_kernel
from coll_models_v2.projections import (
    _legendre_nodes,
    conditional_energy_mean_map,
    energy_quantiles,
    fit_angular_projection,
    fit_energy_projection,
    incoming_partition_density,
)
from dsmc_v2_contracts import cell_invariants


def _beta_angular_moments(p: float, q: float) -> tuple[float, float]:
    total = p + q
    mean_w = p / total
    second_w = p * (p + 1.0) / (total * (total + 1.0))
    mean_c = 2.0 * mean_w - 1.0
    second_c = 4.0 * second_w - 4.0 * mean_w + 1.0
    return mean_c, 0.5 * (3.0 * second_c - 1.0)


@pytest.mark.parametrize("p,q", [
    (1.0, 1.0), (2.0, 1.0), (0.7, 1.0), (3.0, 5.0), (1.4142, 1.0),
])
def test_t1_exact_angular_projection_matches_beta_moments(p, q):
    target = _beta_angular_moments(p, q)
    fitted = fit_angular_projection(*target)
    np.testing.assert_allclose(fitted.moments, target, atol=1.0e-8)


def test_t2_exact_projection_reproduces_both_vss_moments():
    target = _beta_angular_moments(1.7, 1.0)
    fitted = fit_angular_projection(*target)
    np.testing.assert_allclose(fitted.moments, target, atol=1.0e-8)


def test_t3_exact_projection_has_no_vss_lambda2_floor():
    fitted = fit_angular_projection(0.0, -0.05)
    np.testing.assert_allclose(fitted.moments, [0.0, -0.05], atol=1.0e-8)


@pytest.mark.parametrize("mean,second,p_exchange", [
    (0.50, 0.30, 0.40), (0.30, 0.13, 0.20), (0.72, 0.56, 0.65),
])
def test_t4_direct_exchange_probability_and_reset_moments(mean, second, p_exchange):
    rng = np.random.default_rng(20260901)
    count = 150_000
    z_in = rng.beta(2.0, 2.0, count)
    projection = fit_energy_projection(mean, second)
    probability = np.linspace(0.0, 1.0, 8193)
    quantile = energy_quantiles(projection.parameters, probability)
    opened = rng.random(count) < p_exchange
    z_out = z_in.copy()
    z_out[opened] = np.interp(rng.random(np.count_nonzero(opened)), probability, quantile)
    # Pinned to the conditional form: this test is about *recovering* an
    # arbitrary invariant law from data.  The bridge cannot do that by
    # construction -- it imposes Beta(2,2) -- which is the subject of
    # test_bridge_kernel.py::test_bridge_imposes_its_reference_law.
    fitted = fit_exchange_kernel(z_in, z_out, np.ones(count), model_form=False,
                                 kernel_form="conditional_iprojection_v2")
    assert fitted["p_exch"] == pytest.approx(p_exchange, abs=0.015)
    # The invariant law of a gated generator is recovered, but only
    # approximately: the atom the generator has is not a member of the fitted
    # family. Measured error is 0.001 at p=0.4, 0.012 at p=0.65 and 0.028 at
    # p=0.2, the amplification being the shrinking fraction of collisions that
    # carry information about the target law.
    assert fitted["reset_mean"] == pytest.approx(mean, abs=0.03)
    assert fitted["reset_second_moment"] == pytest.approx(second, abs=0.03)


@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0, 2.0])
def test_t5_collision_pool_partition_places_theta_fixed_point(theta):
    target = _energy_weighted_partition(theta)
    assert target / (1.0 - target) == pytest.approx(theta, rel=2.0e-12)


def test_t5b_equal_temperatures_give_beta22_mean():
    assert _incoming_partition_mean(1.0) == pytest.approx(0.5, abs=1.0e-12)


def _memoryless_nodes(parameters, mean_loss, lambda3=0.0, lambda4=0.0):
    """Constant-parameter theta sweep in the shape ``_stability_rows`` expects."""
    return [{
        "alpha": 0.8,
        "theta": float(theta),
        "aspect_ratio": 2.0,
        "energy": {"p_exch": 0.4,
                   "lambda1": float(parameters[0]),
                   "lambda2": float(parameters[1]),
                   "lambda3": float(lambda3),
                   "lambda4": float(lambda4)},
        "uncertainty": {"mean_partition_out": {"standard_error": 1.0e-5}},
    } for theta in np.linspace(0.1, 3.0, 13)]


def _post_collision_partition(theta, parameters, offset=0.0):
    """Independent replica of the module's E[z_out] for the fitted kernel."""
    grid, quadrature = _legendre_nodes(192, 0.0, 1.0)
    mass = incoming_partition_density(theta, grid) * quadrature
    mass = mass / np.sum(mass)
    return float(mass @ conditional_energy_mean_map(parameters, grid, offset=offset))


@pytest.mark.parametrize("root,mean_loss", [(0.5, 0.02), (1.0, 0.05), (2.0, 0.08)])
def test_t6_composed_surface_stability_is_numerical_and_negative(root, mean_loss):
    incoming = _energy_weighted_partition(root)
    post_at_root = (incoming - root * mean_loss / (2.0 / 3.0 + root)) / (1.0 - mean_loss)
    # A memoryless kernel has a theta-independent post-collision partition, so
    # placing its mean at post_at_root places the drift root at root exactly.
    projection = fit_energy_projection(post_at_root,
                                       post_at_root**2
                                       + 0.5 * post_at_root * (1.0 - post_at_root))
    parameters = np.array([projection.parameters[0], projection.parameters[1], 0.0])
    assert _post_collision_partition(root, parameters) == pytest.approx(post_at_root, abs=1e-9)

    class Loss:
        @staticmethod
        def parameters(alpha, aspect_ratio):
            return {"mean_loss_fraction": mean_loss}

    row = _stability_rows(_memoryless_nodes(projection.parameters, mean_loss), Loss())[0]
    assert row["roots"][0] == pytest.approx(root, abs=2.0e-7)
    assert row["drift_derivative"] < 0.0
    assert row["unique_stable"]
    assert row["mean_scalar_loss"] == mean_loss
    assert row["includes_surface_derivatives"]


@pytest.mark.parametrize("root,mean_loss,lambda3", [(0.6, 0.04, 4.0), (1.4, 0.06, 12.0)])
def test_t6b_memory_term_enters_the_theta_fixed_point(root, mean_loss, lambda3):
    """The drift must see the memory parameter, not only the marginal tilt."""
    incoming = _energy_weighted_partition(root)
    post_at_root = (incoming - root * mean_loss / (2.0 / 3.0 + root)) / (1.0 - mean_loss)

    def residual(lambda1):
        return _post_collision_partition(
            root, np.array([lambda1, -0.5, lambda3])) - post_at_root

    lambda1 = brentq(residual, -60.0, 60.0, xtol=1.0e-13)

    class Loss:
        @staticmethod
        def parameters(alpha, aspect_ratio):
            return {"mean_loss_fraction": mean_loss}

    nodes = _memoryless_nodes(np.array([lambda1, -0.5]), mean_loss, lambda3=lambda3)
    row = _stability_rows(nodes, Loss())[0]
    assert row["roots"] == pytest.approx([root], abs=1.0e-5)
    assert row["drift_derivative"] < 0.0
    # Dropping the memory term moves the root, so it is genuinely load bearing.
    without = _stability_rows(
        _memoryless_nodes(np.array([lambda1, -0.5]), mean_loss, lambda3=0.0), Loss())[0]
    assert not without["roots"] or abs(without["roots"][0] - root) > 0.05


def _a2_tr_of_s(s):
    return 0.6 * gamma(7.0 / s) * gamma(3.0 / s) / gamma(5.0 / s) ** 2 - 1.0


def _a2_rot_of_s(s):
    return 0.5 * gamma(6.0 / s) * gamma(2.0 / s) / gamma(4.0 / s) ** 2 - 1.0


@pytest.mark.parametrize("target", [0.20, 0.05, -0.05, -0.15])
def test_t7_stretched_translational_cumulants(target):
    rng = np.random.default_rng(20260901)
    s = brentq(lambda value: _a2_tr_of_s(value) - target, 0.35, 40.0)
    radius = rng.gamma(3.0 / s, 1.0, 300_000) ** (1.0 / s)
    radius *= np.sqrt(1.5 / np.mean(radius**2))
    measured = 4.0 * np.mean(radius**4) / 15.0 - 1.0
    assert measured == pytest.approx(target, abs=0.012)


@pytest.mark.parametrize("target", [0.15, -0.10])
def test_t7b_stretched_rotational_cumulants(target):
    rng = np.random.default_rng(20260901)
    s = brentq(lambda value: _a2_rot_of_s(value) - target, 0.35, 40.0)
    radius = rng.gamma(2.0 / s, 1.0, 300_000) ** (1.0 / s)
    radius *= np.sqrt(1.0 / np.mean(radius**2))
    measured = 0.5 * np.mean(radius**4) - 1.0
    assert measured == pytest.approx(target, abs=0.012)


def _random_axes(count, rng):
    axes = rng.normal(size=(count, 3))
    return axes / np.linalg.norm(axes, axis=1)[:, None]


def test_t8_acu_excitation_breaks_factorised_collinearity():
    rng = np.random.default_rng(20260901)
    count = 250_000
    velocity = rng.normal(size=(count, 3))
    velocity *= np.sqrt(1.5 / np.mean(np.sum(velocity**2, axis=1)))
    chat = velocity / np.linalg.norm(velocity, axis=1)[:, None]
    grid = np.linspace(-1.0, 1.0, 4001)
    density = np.exp(2.5 * grid**2)
    cdf = np.cumsum(density); cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    cosine = np.interp(rng.random(count), cdf, grid)
    trial = np.tile([1.0, 0.0, 0.0], (count, 1))
    trial[np.abs(chat[:, 0]) > 0.9] = [0.0, 1.0, 0.0]
    e1 = np.cross(chat, trial); e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(chat, e1)
    phi = 2.0 * np.pi * rng.random(count)
    axes = (cosine[:, None] * chat
            + (np.sqrt(1.0 - cosine**2) * np.cos(phi))[:, None] * e1
            + (np.sqrt(1.0 - cosine**2) * np.sin(phi))[:, None] * e2)
    tangent = np.cross(axes, _random_axes(count, rng))
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    omega = tangent * rng.rayleigh(size=count)[:, None]
    features, _ = cell_invariants(velocity, omega, axes)
    assert abs(features[4]) < 0.004
    assert abs(features[5]) < 0.004
    assert abs(features[3]) > 0.05


def test_t9_factorised_acu_identity_uses_production_normalisation():
    rng = np.random.default_rng(20260901)
    count = 300_000
    covariance = 0.5 * (np.eye(3) + np.diag([0.30, -0.10, -0.20]))
    velocity = rng.normal(size=(count, 3)) @ np.linalg.cholesky(covariance).T
    axes = _random_axes(count, rng); axes[:, 2] *= 1.6
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    tangent = np.cross(axes, _random_axes(count, rng))
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    omega = tangent * rng.rayleigh(size=count)[:, None]
    features, _ = cell_invariants(velocity, omega, axes)
    assert features[3] == pytest.approx(4.0 * features[7] / 3.0, abs=0.006)


def test_t10_irreducible_spin_anisotropy_vanishes_for_isotropic_tangent_spin():
    rng = np.random.default_rng(20260901)
    count = 250_000
    axes = _random_axes(count, rng); axes[:, 2] *= 2.0
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    trial = np.tile([1.0, 0.0, 0.0], (count, 1))
    trial[np.abs(axes[:, 0]) > 0.9] = [0.0, 1.0, 0.0]
    e1 = np.cross(axes, trial); e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(axes, e1)
    omega = (rng.normal(size=count)[:, None] * e1
             + rng.normal(size=count)[:, None] * e2) / np.sqrt(2.0)
    features, _ = cell_invariants(rng.normal(size=(count, 3)), omega, axes)
    assert abs(features[6]) < 0.004


def test_t11_tensor_u_statistic_removes_isotropic_noise_floor():
    rng = np.random.default_rng(20260901)
    replicas, count = 5000, 40
    c = rng.normal(size=(replicas, count, 3)) / np.sqrt(2.0)
    square = np.einsum("rni,rni->rn", c, c)
    particle = 2.0 * (np.einsum("rni,rnj->rnij", c, c)
                      - square[..., None, None] * np.eye(3) / 3.0)
    total = np.sum(particle, axis=1)
    naive = np.einsum("rij,rij->r", total, total) / count**2
    self_term = np.einsum("rnij,rnij->r", particle, particle)
    unbiased = (np.einsum("rij,rij->r", total, total) - self_term) / (count * (count - 1))
    assert abs(np.mean(unbiased)) < 0.15 * np.mean(naive)
