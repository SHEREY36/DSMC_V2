"""Gates that were testing the wrong thing, and what they test now.

Three of the release gates could not do their job as written. The elastic gate
compared a weakly identified quantity against a flat tolerance, so it called a
well resolved node a failure and would have called a badly resolved one a pass.
The incoming-partition check in the implementation plan used the ratio of the
means where the mean of the ratio is meant, which is wrong by 28 percent at
theta = 0.2. And a pure significance test on the hit propensity necessarily
tightens as the event count grows, so it cannot express a physical tolerance.
"""

import numpy as np
import pytest

from coll_models_v2.projections import _legendre_nodes, incoming_partition_density
from coll_models_v2.weights import RELATIVE_BIAS_TOLERANCE


def _exact_partition_mean(theta, quadrature=256):
    nodes, quad = _legendre_nodes(quadrature, 0.0, 1.0)
    mass = incoming_partition_density(theta, nodes) * quad
    return float((mass / mass.sum()) @ nodes)


@pytest.mark.parametrize("theta", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
def test_incoming_partition_law_matches_the_two_gamma_ratio(theta):
    """Both modal energies are Gamma(2), so z = X/(X+Y) has a closed form."""
    rng = np.random.default_rng(int(1000 * theta))
    x = rng.gamma(2.0, theta, 400000)
    y = rng.gamma(2.0, 1.0, 400000)
    assert _exact_partition_mean(theta) == pytest.approx(
        float(np.mean(x / (x + y))), abs=2.0e-3)


@pytest.mark.parametrize("theta,gap", [(0.2, 0.20), (2.0, 0.02)])
def test_ratio_of_means_is_not_the_mean_of_the_ratio(theta, gap):
    """The plan's `theta/(theta+1)` gate would fail a correct sampler.

    At theta = 0.2 it is low by 22 percent of the true value, which is forty
    times the tolerance the gate is written with.
    """
    exact = _exact_partition_mean(theta)
    naive = theta / (theta + 1.0)
    assert abs(naive - exact) / exact > gap
    assert abs(naive - exact) / exact > RELATIVE_BIAS_TOLERANCE


def _elastic_gate(value, target, standard_error):
    """The deployed rule: three standard errors or a flat 2 percent, looser wins."""
    allowed = max(3.0 * standard_error, RELATIVE_BIAS_TOLERANCE) \
        if standard_error is not None else RELATIVE_BIAS_TOLERANCE
    return abs(value - target) <= allowed


def test_elastic_gate_passes_a_well_resolved_marginal_node():
    # (alpha=1, theta=0.2, AR=2): 0.3222 against 0.3000 with a bootstrap error
    # of 0.0111, so 2.0 sigma. A flat 0.02 tolerance calls this a failure.
    assert not abs(0.3222 - 0.3) <= 0.02
    assert _elastic_gate(0.3222, 0.3, 0.0111)


def test_elastic_gate_still_fails_a_genuinely_unidentified_node():
    # (alpha=1, theta=0.2, AR=1.1): 0.5811 against 0.3000 with an error of
    # 0.0741, so 3.8 sigma. Z_rot is 16.5 there, meaning the kernel is nearly
    # the identity and its invariant law is inferred from a small residue.
    assert not _elastic_gate(0.5811, 0.3, 0.0741)


def test_elastic_gate_does_not_tighten_without_bound():
    """A pure significance test would reject a physically negligible offset."""
    assert _elastic_gate(0.5 + 0.5 * RELATIVE_BIAS_TOLERANCE, 0.5, 1.0e-6)
