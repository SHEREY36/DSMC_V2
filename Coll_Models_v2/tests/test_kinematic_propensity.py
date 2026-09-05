"""Properties of the kinematic acceptance propensity.

The generator accepts a proposal when two turning spherocylinders touch during
the encounter, so its acceptance is the dynamic excluded area rather than the
static shadow. These tests pin the four things that makes it a usable
Radon-Nikodym weight: it reduces to the analytic shadow when the rods do not
turn, it is a scalar of the geometry rather than of the frame, it does not
depend on the arbitrary staging distance, and rotation can only enlarge it.
"""

import numpy as np
import pytest

from coll_models_v2.weights import (
    DEFAULT_OFFSETS,
    debiased_inverse,
    encounter_propensity,
    projected_excluded_area,
)

DIAMETER, LENGTH = 1.0, 2.0
STAGING = 1.01 * (LENGTH + DIAMETER)


def _sample(count, spin_scale, seed=1):
    rng = np.random.default_rng(seed)
    director = []
    spin = []
    for _ in range(2):
        u = rng.standard_normal((count, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        director.append(u)
        w = rng.standard_normal((count, 3))
        w -= np.einsum("ni,ni->n", w, u)[:, None] * u      # torque-free: omega ⊥ u
        spin.append(spin_scale * w)
    ghat = rng.standard_normal((count, 3))
    ghat /= np.linalg.norm(ghat, axis=1, keepdims=True)
    return ghat, np.ones(count), director, spin


def _analytic_shadow(ghat, director):
    first = np.linalg.norm(np.cross(director[0], ghat), axis=1)
    second = np.linalg.norm(np.cross(director[1], ghat), axis=1)
    triple = np.abs(np.einsum("ni,ni->n", ghat, np.cross(director[0], director[1])))
    return (np.pi * DIAMETER ** 2 + 2.0 * DIAMETER * LENGTH * (first + second)
            + LENGTH ** 2 * triple)


def test_non_rotating_encounter_reproduces_the_analytic_shadow():
    """The static area is the zero-spin limit, so this pins the whole geometry."""
    ghat, speed, director, _ = _sample(2500, 0.0)
    propensity = encounter_propensity(ghat, speed, director, [np.zeros((2500, 3))] * 2,
                                      DIAMETER, LENGTH, STAGING, offsets=512)
    exact = _analytic_shadow(ghat, director) / (4.0 * STAGING ** 2)
    assert np.mean(propensity / exact) == pytest.approx(1.0, abs=2.0e-3)


def test_propensity_is_invariant_under_a_global_rotation():
    """The effective area is a scalar of the geometry, so the mean must not move.

    Per-event equality is not expected and is not claimed: the lattice is laid
    out in an arbitrary basis transverse to ghat, so a rotated frame is a
    different Monte-Carlo realisation of the same integral. The estimator is
    unbiased for any such basis, which is what this pins.
    """
    ghat, speed, director, spin = _sample(1200, 0.7)
    rotation = np.linalg.qr(np.random.default_rng(5).standard_normal((3, 3)))[0]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    plain = encounter_propensity(ghat, speed, director, spin,
                                 DIAMETER, LENGTH, STAGING, offsets=128)
    turned = encounter_propensity(
        ghat @ rotation.T, speed, [u @ rotation.T for u in director],
        [w @ rotation.T for w in spin], DIAMETER, LENGTH, STAGING, offsets=128)
    assert np.mean(turned) == pytest.approx(np.mean(plain), rel=5.0e-3)
    # and the per-event scatter is the quadrature error, so it shrinks with M
    coarse = np.std(turned - plain)
    fine = np.std(
        encounter_propensity(ghat @ rotation.T, speed, [u @ rotation.T for u in director],
                             [w @ rotation.T for w in spin], DIAMETER, LENGTH,
                             STAGING, offsets=384)
        - encounter_propensity(ghat, speed, director, spin,
                               DIAMETER, LENGTH, STAGING, offsets=384))
    assert fine < 0.8 * coarse


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_propensity_does_not_depend_on_the_staging_distance(factor):
    """Contact needs a centre separation below L + D, so the run-up is inert.

    This is the claim that separates real physics from a setup artefact: a
    cross-section that grew with the staging distance would be an artefact.
    """
    ghat, speed, director, spin = _sample(1600, 0.7)
    reference = encounter_propensity(ghat, speed, director, spin,
                                     DIAMETER, LENGTH, STAGING, offsets=160)
    scaled = factor ** 2 * encounter_propensity(ghat, speed, director, spin,
                                                DIAMETER, LENGTH, factor * STAGING,
                                                offsets=160)
    # The propensity is per unit impact box, which grows with the staging
    # distance; the effective area it implies is what must be invariant. A
    # longer run-up leaves each pair at a different orientation on entry, so
    # only the mean is preserved -- test it against its own sampling error
    # rather than a fixed tolerance.
    difference = float(np.mean(scaled) - np.mean(reference))
    error = float(np.sqrt((np.var(scaled, ddof=1) + np.var(reference, ddof=1))
                          / len(reference)))
    assert abs(difference) <= 3.0 * error
    assert abs(difference) <= 0.02 * float(np.mean(reference))


def test_rotation_can_only_enlarge_the_effective_area():
    ghat, speed, director, spin = _sample(1000, 0.7)
    still = encounter_propensity(ghat, speed, director, [np.zeros_like(w) for w in spin],
                                 DIAMETER, LENGTH, STAGING, offsets=160)
    for scale in (1.0, 3.0):
        turning = encounter_propensity(ghat, speed, director, [scale * w for w in spin],
                                       DIAMETER, LENGTH, STAGING, offsets=160)
        assert np.mean(turning) > np.mean(still)


def test_debiased_inverse_removes_the_leading_monte_carlo_inflation():
    rng = np.random.default_rng(3)
    offsets, truth = 128, 0.25
    draws = rng.binomial(offsets, truth, 400000) / offsets
    draws = draws[draws > 0.0]
    plain = float(np.mean(1.0 / draws))
    corrected = float(np.mean(debiased_inverse(draws, offsets)))
    assert plain > 1.0 / truth * 1.01                 # the bias is real
    assert corrected == pytest.approx(1.0 / truth, rel=3.0e-3)


def test_debiased_inverse_rejects_a_zero_propensity():
    with pytest.raises(ValueError, match="positive"):
        debiased_inverse(np.array([0.3, 0.0]), DEFAULT_OFFSETS)


def test_staging_below_the_contact_reach_is_refused():
    ghat, speed, director, spin = _sample(10, 1.0)
    with pytest.raises(ValueError, match="staging"):
        encounter_propensity(ghat, speed, director, spin, DIAMETER, LENGTH,
                             0.5 * (LENGTH + DIAMETER), offsets=16)
