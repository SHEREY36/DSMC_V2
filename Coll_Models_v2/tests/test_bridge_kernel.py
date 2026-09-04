"""Properties of the Sinkhorn-bridge exchange kernel.

The conditional I-projection *infers* the invariant law from a kernel that, at
weak coupling, is close to the identity; the sentinel showed it landing 1 to 3
percent -- and at one node 29 percent -- away from equipartition where the
elastic physics demands it exactly.  The bridge instead makes the kernel
reversible with respect to Beta(2,2) by construction, so equipartition holds
identically for any memory, and only the dissipative tilt can move it.

These tests pin that construction: exact invariance, detailed balance,
normalisation, that the constraint is inert when a tilt is present, and that
the warm-started bracket used by bootstrap replicates keeps lambda3 free.
"""

import numpy as np
import pytest

from coll_models_v2.fit_exchange import (
    KERNEL_FORMS,
    WARM_BRACKET,
    fit_bridge_kernel,
    fit_exchange_kernel,
)
from coll_models_v2.projections import (
    _legendre_nodes,
    bridge_logpdf,
    bridge_potential,
    bridge_stationary,
)


MEMORIES = [0.0, 2.0, 6.0, 50.0, 220.0, 600.0]


@pytest.mark.parametrize("memory", MEMORIES)
def test_untilted_bridge_is_exactly_equipartition(memory):
    """The whole point: the invariant law is Beta(2,2) for *any* memory."""
    nodes, mass = bridge_stationary(np.array([memory]))
    assert mass @ nodes == pytest.approx(0.5, abs=1.0e-9)
    assert mass @ (nodes * nodes) == pytest.approx(0.3, abs=1.0e-9)


@pytest.mark.parametrize("memory", MEMORIES)
def test_untilted_bridge_satisfies_detailed_balance(memory):
    """pi(z) p(z'|z) == pi(z') p(z|z') with pi = Beta(2,2)."""
    z = np.array([0.08, 0.23, 0.5, 0.77, 0.95])
    grid_in, grid_out = (x.ravel() for x in np.meshgrid(z, z, indexing="ij"))
    log_pi = np.log(6.0 * z * (1.0 - z))
    forward = bridge_logpdf(np.array([memory]), grid_in, grid_out)
    reverse = bridge_logpdf(np.array([memory]), grid_out, grid_in)
    flux = forward + np.repeat(log_pi, len(z))
    reverse_flux = reverse + np.tile(log_pi, len(z))
    assert np.max(np.abs(flux - reverse_flux)) < 1.0e-8


@pytest.mark.parametrize("memory", MEMORIES)
def test_bridge_potential_has_the_expected_shape(memory):
    """h is flat with no coupling, and otherwise decreasing across a fixed span.

    log h(0) - log h(1) is exactly lambda3 / 2; on a Gauss-Legendre grid, whose
    nodes stop short of both ends, that becomes lambda3 (z_last - z_first) / 2.
    The identity holds to machine precision and is a cheap check that Sinkhorn
    actually converged -- a stalled or truncated iteration misses it at once.
    h is *not* symmetric under z -> 1-z: the coupling exp(lambda3 z z') is not.
    """
    nodes, _ = _legendre_nodes(256, 0.0, 1.0)
    potential = np.asarray(bridge_potential(memory))
    assert np.all(np.isfinite(potential))
    assert np.all(np.diff(potential) <= 1.0e-12)
    span = 0.5 * memory * (nodes[-1] - nodes[0])
    assert np.ptp(potential) == pytest.approx(span, abs=1.0e-9)


@pytest.mark.parametrize("parameters", [
    np.array([0.0]),
    np.array([33.9]),
    np.array([220.0]),
    np.array([12.0, 1.5, -2.0]),
    np.array([38.2, 12.1, -12.4, -0.6]),
])
def test_bridge_conditional_integrates_to_one(parameters):
    nodes, weights = _legendre_nodes(256, 0.0, 1.0)
    loss = None if len(parameters) < 4 else np.full(len(nodes), 0.21)
    for z in (0.05, 0.3, 0.5, 0.82, 0.97):
        z_in = np.full(len(nodes), z)
        density = np.exp(bridge_logpdf(parameters, z_in, nodes, loss))
        assert weights @ density == pytest.approx(1.0, abs=1.0e-7)


def _bridge_sample(memory, count, seed, tilt=None):
    """Draw z' | z from the bridge by inverse-CDF on the quadrature grid."""
    rng = np.random.default_rng(seed)
    nodes, weights = _legendre_nodes(256, 0.0, 1.0)
    parameters = np.array([memory]) if tilt is None else np.append(memory, tilt)
    z_in = rng.beta(2.0, 2.0, count)
    z_out = np.empty(count)
    for start in range(0, count, 4096):
        block = slice(start, min(start + 4096, count))
        rows = np.exp(bridge_logpdf(
            parameters,
            np.repeat(z_in[block], len(nodes)),
            np.tile(nodes, len(z_in[block])))).reshape(-1, len(nodes)) * weights
        cumulative = np.cumsum(rows, axis=1)
        cumulative /= cumulative[:, -1:]
        draw = rng.random((cumulative.shape[0], 1))
        z_out[block] = nodes[np.argmax(cumulative > draw, axis=1)]
    return z_in, z_out


def test_elastic_block_recovers_memory_and_pins_equipartition():
    z_in, z_out = _bridge_sample(20.0, 12_000, seed=7)
    weight = np.ones(len(z_in))
    fit = fit_bridge_kernel(z_in, z_out, weight)
    assert fit["elastic_block"] is True
    assert fit["lambda3"] == pytest.approx(20.0, rel=0.15)
    assert fit["stationary_mean"] == pytest.approx(0.5, abs=1.0e-9)
    assert fit["stationary_second_moment"] == pytest.approx(0.3, abs=1.0e-9)


def test_dissipative_tilt_moves_the_fixed_point_off_equipartition():
    """The constraint must bind only elastically, or it would be wrong physics."""
    z_in, z_out = _bridge_sample(20.0, 12_000, seed=11, tilt=[6.0, -8.0])
    loss = np.full(len(z_in), 0.3)
    fit = fit_bridge_kernel(z_in, z_out, np.ones(len(z_in)), loss=loss)
    assert fit["elastic_block"] is False
    assert abs(fit["stationary_mean"] - 0.5) > 0.01
    assert fit["projection_residual"] < 1.0e-8


def test_warm_bracket_keeps_memory_free_but_bounded():
    z_in, z_out = _bridge_sample(20.0, 8_000, seed=3)
    weight = np.ones(len(z_in))
    cold = fit_bridge_kernel(z_in, z_out, weight)
    anchor = np.array([cold["lambda3"]])
    rng = np.random.default_rng(5)
    spread = []
    for _ in range(3):
        pick = rng.integers(0, len(z_in), len(z_in))
        warm = fit_bridge_kernel(z_in[pick], z_out[pick], weight[pick], initial=anchor)
        # inside the bracket ...
        assert WARM_BRACKET[0] * cold["lambda3"] <= warm["lambda3"] \
            <= WARM_BRACKET[1] * cold["lambda3"]
        assert warm["stationary_mean"] == pytest.approx(0.5, abs=1.0e-9)
        spread.append(warm["lambda3"])
    # ... but genuinely resampled, not pinned to the anchor.
    assert np.std(spread) > 0.0


def test_dispatch_serves_the_same_contract_as_the_conditional_kernel():
    z_in, z_out = _bridge_sample(15.0, 5_000, seed=13)
    weight = np.ones(len(z_in))
    loss = np.linspace(0.05, 0.4, len(z_in))
    bridge = fit_exchange_kernel(z_in, z_out, weight, loss=loss,
                                 kernel_form="sinkhorn_bridge_v2")
    conditional = fit_exchange_kernel(z_in, z_out, weight, loss=loss,
                                      kernel_form="conditional_iprojection_v2")
    assert set(conditional).issubset(set(bridge))
    assert bridge["kernel_form"] == "sinkhorn_bridge_v2"
    assert bridge["reset_mean"] == bridge["stationary_mean"]


def test_default_kernel_form_is_the_bridge():
    z_in, z_out = _bridge_sample(15.0, 4_000, seed=17)
    fit = fit_exchange_kernel(z_in, z_out, np.ones(len(z_in)), model_form=False)
    assert fit["kernel_form"] == "sinkhorn_bridge_v2"
    assert "sinkhorn_bridge_v2" in KERNEL_FORMS


def test_unknown_kernel_form_is_rejected():
    z_in, z_out = _bridge_sample(5.0, 500, seed=19)
    with pytest.raises(ValueError, match="unknown kernel_form"):
        fit_exchange_kernel(z_in, z_out, np.ones(len(z_in)), kernel_form="beta22_gate")


def test_bridge_imposes_its_reference_law_even_when_the_data_disagree():
    """The bridge's one real cost, made explicit.

    Data whose invariant partition is Beta(3,4) -- mean 3/7, not 1/2 -- is fit
    by both forms.  The conditional form recovers 3/7; the bridge reports 1/2,
    because Beta(2,2) reversibility is built in rather than inferred.  The
    bridge is therefore only correct insofar as Beta(2,2) really is this
    system's equilibrium, which for the CTC generator is independently known:
    incoming states are drawn from z(1-z)(z/theta + 1-z)^-4, and the measured
    elastic sentinel node at theta=1, AR=1.1 has mean 0.4990 and variance
    0.0502 against Beta(2,2)'s 0.5 and 0.05.

    If that reference were ever wrong, this test is where it would show.
    """
    rng = np.random.default_rng(83)
    count = 120_000
    z_in = rng.beta(2.0, 2.0, count)
    opened = rng.random(count) < 0.37
    z_out = z_in.copy()
    z_out[opened] = rng.beta(3.0, 4.0, np.count_nonzero(opened))
    weight = np.ones(count)

    free = fit_exchange_kernel(z_in, z_out, weight, model_form=False,
                               kernel_form="conditional_iprojection_v2")
    bridge = fit_exchange_kernel(z_in, z_out, weight, model_form=False,
                                 kernel_form="sinkhorn_bridge_v2")
    assert free["reset_mean"] == pytest.approx(3.0 / 7.0, abs=0.006)
    assert bridge["reset_mean"] == pytest.approx(0.5, abs=1.0e-9)
    assert bridge["reset_second_moment"] == pytest.approx(0.3, abs=1.0e-9)
