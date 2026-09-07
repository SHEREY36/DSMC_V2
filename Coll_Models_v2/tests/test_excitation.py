"""Reweighted excitation: does it reach the invariants, and does it stay honest?"""

from types import SimpleNamespace

import numpy as np
import pytest

from coll_models_v2.excitation import (
    MINIMUM_ESS, ONE_SIDED, UNIDENTIFIABLE, _normal_score, excite,
)
from dsmc_v2_contracts.io import AI, ATTEMPT_DTYPE


def _maxwellian_attempt_run(count=20000):
    rng = np.random.default_rng(8128)
    attempts = np.zeros(count, dtype=ATTEMPT_DTYPE)
    values = attempts["values"]
    for particle in (1, 2):
        c = rng.normal(size=(count, 3))
        u = rng.normal(size=(count, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        w = rng.normal(size=(count, 3))
        w -= np.einsum("ni,ni->n", w, u)[:, None] * u
        for prefix, vector in ((f"c{particle}", c),
                               (f"omega{particle}", w),
                               (f"u{particle}", u)):
            for component, axis in enumerate("xyz"):
                values[:, AI[f"{prefix}_{axis}"]] = vector[:, component]
    return SimpleNamespace(
        attempts=attempts,
        metadata={"velocity_scale": np.sqrt(2.0), "omega_scale": np.sqrt(2.0),
                  "mass": 1.0, "moi_perpendicular": 1.0})


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


def test_mean_spin_is_not_confused_with_forbidden_axial_spin():
    """W2=|<omega>|^2 can vary even though every smooth rod has omega.u=0."""
    assert "W2" not in UNIDENTIFIABLE


def test_quadratic_invariants_are_flagged_one_sided():
    """PiPi, QQ and RtRt are contractions: non-negative, so only one sign of
    the achieved shift exists and a two-sided design would be misread."""
    for name in ("PiPi", "QQ", "RtRt", "qtr2", "qrot2", "W2"):
        assert name in ONE_SIDED


def test_corrected_scores_reach_scalar_spin_and_signed_cross_invariants():
    run = _maxwellian_attempt_run()
    negative_acu = excite(run, "A_cu", -0.4)
    positive_acu = excite(run, "A_cu", 0.4)
    spin = excite(run, "W2", 0.4)
    parallel = excite(run, "PiQ", 0.4)
    opposed = excite(run, "PiQ_opposed", 0.4)
    assert negative_acu["features"]["A_cu"] < -0.1
    assert positive_acu["features"]["A_cu"] > 0.1
    assert spin["features"]["W2"] > 0.02
    assert parallel["features"]["PiQ"] > 0.005
    assert opposed["features"]["PiQ"] < -0.005
    assert min(row["ess_fraction"] for row in
               (negative_acu, positive_acu, spin, parallel, opposed)) > MINIMUM_ESS
