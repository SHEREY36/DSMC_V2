"""The I-projection solver must converge across the whole feasible region.

A sentinel node was discarded with "angular I-projection residual 1.360e-08":
the quasi-Newton search stalled just above the 1e-8 convergence threshold, and
because a raised exception fails every gate on a node, one marginal solve took
down the energy kernel, the measure diagnostics and the elastic limit with it.
"""

import numpy as np
import pytest

from coll_models_v2.projections import (
    PROJECTION_TOLERANCE,
    angular_moments_are_feasible,
    fit_angular_projection,
    fit_energy_projection,
)


def _feasible_angular_targets(count=24):
    for first in np.linspace(-0.85, 0.85, count):
        m = 0.5 * (1.0 + first)
        floor = 6.0 * m * m - 6.0 * m + 1.0
        for fraction in np.linspace(0.02, 0.98, count):
            yield float(first), float(floor + fraction * (1.0 - floor))


def test_angular_projection_converges_across_the_feasible_region():
    worst, count = 0.0, 0
    for first, second in _feasible_angular_targets():
        fit = fit_angular_projection(first, second)
        assert fit.converged, (first, second, fit.residual)
        worst, count = max(worst, fit.residual), count + 1
    assert count > 500
    assert worst < 1.0e-8


@pytest.mark.parametrize("first,second", [(-0.6, -0.4), (0.6, -0.4), (0.0, -0.9)])
def test_infeasible_angular_moments_say_so(first, second):
    """Outside the moment cone the message must name the cause, not a residual."""
    assert not angular_moments_are_feasible(first, second)
    with pytest.raises(ValueError, match="outside the moment cone"):
        fit_angular_projection(first, second)


def test_energy_projection_converges_across_the_feasible_region():
    worst = 0.0
    for mean in np.linspace(0.05, 0.95, 19):
        for fraction in np.linspace(0.05, 0.95, 19):
            second = mean * mean + fraction * (mean - mean * mean)
            fit = fit_energy_projection(float(mean), float(second))
            assert fit.converged, (mean, second, fit.residual)
            worst = max(worst, fit.residual)
    assert worst < 1.0e-8


def test_estimators_do_not_reject_what_the_release_gate_accepts():
    """The raise threshold and the QA gate must be the same number."""
    assert PROJECTION_TOLERANCE == 1.0e-6
