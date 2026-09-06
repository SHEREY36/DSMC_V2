"""Reweighted excitation: does it reach the invariants, and does it stay honest?"""

import numpy as np
import pytest

from coll_models_v2.excitation import (
    MINIMUM_ESS, ONE_SIDED, UNIDENTIFIABLE, _normal_score,
)


def test_normal_score_is_monotone_and_light_tailed():
    """The whole reason for the transform.

    Tilting on raw kurtosis, exp(-c^2 + eta c^4), is non-normalisable for every
    positive eta: the target does not exist and finite-sample weights collapse
    onto the largest event. The rank transform keeps the ordering and bounds the
    tails, so both directions stay conditioned.
    """
    rng = np.random.default_rng(0)
    heavy = rng.standard_normal(20000) ** 4
    score = _normal_score(heavy)
    assert np.all(np.diff(score[np.argsort(heavy)]) >= 0.0)
    assert abs(score.mean()) < 0.02 and abs(score.std() - 1.0) < 0.02

    for eta in (0.3, 0.6):
        for stat, tag in ((heavy, "raw"), (score, "rank")):
            s = (stat - stat.mean()) / stat.std()
            w = np.exp(eta * s - (eta * s).max())
            ess = float(w.sum() ** 2 / np.sum(w * w)) / len(w)
            if tag == "rank":
                assert ess > MINIMUM_ESS
            else:
                assert ess < 0.35        # the failure the transform removes


def test_axial_spin_families_are_declared_unidentifiable():
    """A smooth rod has no spin about its own axis, so W2 cannot be excited."""
    assert "W2" in UNIDENTIFIABLE


def test_quadratic_invariants_are_flagged_one_sided():
    """PiPi, QQ and RtRt are contractions: non-negative, so only one sign of
    the achieved shift exists and a two-sided design would be misread."""
    for name in ("PiPi", "QQ", "RtRt"):
        assert name in ONE_SIDED
