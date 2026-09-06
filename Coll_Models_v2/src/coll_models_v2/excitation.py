"""First-order excitation by reweighting the baseline shard.

The coefficients answer one question: how does the kernel move when the gas
arriving at a collision carries a non-zero invariant? Generating 46 distorted
CTC ensembles per node is one way to ask. The other is importance sampling on
the baseline shard, and under molecular chaos it is exact rather than an
approximation.

    f_eta(x) = r_eta(x) f_0(x)          one-particle tilt
    R_eta(x1, x2) = r_eta(x1) r_eta(x2) pair ratio
    E_eta[F] = E_0[R F] / E_0[R]        for any collision observable F

The contact law and the trajectory conditional on the incoming state are
untouched, so the hit probability cancels from the Radon-Nikodym ratio: fresh
trajectories are not required merely because the incoming distribution changed.

Two things make or break it in practice, and both are measured rather than
assumed:

* the tilt must be applied ONE PARTICLE AT A TIME. Tilting directly on a
  pair-level invariant manufactures correlations between the two partners that
  molecular chaos says are not there.
* the score must have light tails. Tilting on raw kurtosis, exp(-c^2 + eta c^4),
  is non-normalisable for every eta > 0 -- the target distribution does not
  exist, and in finite data the weights collapse onto the largest event. The
  measured effective sample size was exactly zero. A rank-normal score has the
  same ordering with finite weights in both directions.
"""

from __future__ import annotations

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES, cell_invariants
from dsmc_v2_contracts.io import AI, _vec, load_run


MINIMUM_ESS = 0.5
MAXIMUM_WEIGHT_SHARE = 0.01

# Structurally unidentifiable for smooth spherocylinders. A smooth rod carries
# no spin about its own axis -- the generator populates only the two transverse
# components, and the measured axial spin is 4e-13 -- so every invariant built
# on it is identically zero and no reweighting can move it. The design column
# exists but the physics does not.
UNIDENTIFIABLE = ("W2",)

# Non-negative by construction: these are contractions, so both tilt directions
# raise them and only positive amplitudes carry information. That is the
# invariant's own property, not a limitation of the tilt.
ONE_SIDED = ("PiPi", "QQ", "RtRt")


def _normal_score(values: np.ndarray) -> np.ndarray:
    """Rank-to-normal transform: same ordering, tails light enough to tilt."""
    from scipy.special import ndtri
    order = np.argsort(np.argsort(values))
    return ndtri((order + 0.5) / len(values))


def particle_scores(run) -> dict[str, np.ndarray]:
    """Per-particle scores whose tilts move each invariant family.

    Scalar families tilt radial quantities; the tensor and vector families tilt
    a COMPONENT and let the invariant follow, because tilting on the invariant
    itself (which is quadratic) cannot change its sign.
    """
    values = np.asarray(run.attempts["values"])
    c1, c2 = _vec(values, AI, "c1"), _vec(values, AI, "c2")
    w1, w2 = _vec(values, AI, "omega1"), _vec(values, AI, "omega2")
    u1, u2 = _vec(values, AI, "u1"), _vec(values, AI, "u2")
    velocity = np.vstack([c1, c2])
    omega = np.vstack([w1, w2])
    axis = np.vstack([u1, u2])
    peculiar = velocity - velocity.mean(axis=0)
    speed2 = np.sum(peculiar * peculiar, axis=1)
    spin2 = np.sum(omega * omega, axis=1)
    return {
        "a2_tr": speed2,
        "a2_rot": spin2,
        "a11": speed2 * spin2,
        "A_cu": speed2 * np.sqrt(np.maximum(speed2, 0.0)),
        # stress: a component, not the contraction
        "PiPi": peculiar[:, 0] ** 2 - peculiar[:, 1] ** 2,
        # orientational order: the nematic component along z
        "QQ": axis[:, 2] ** 2,
        # spin anisotropy
        "RtRt": omega[:, 0] ** 2 - omega[:, 1] ** 2,
        # heat flux: an odd moment, so the sign of eta is meaningful
        "qtr2": peculiar[:, 0] * speed2,
        "qrot2": peculiar[:, 0] * spin2,
        "W2": np.sum(omega * axis, axis=1) ** 2,
    }, velocity, omega, axis


def excite(run, family: str, eta: float,
           relative_speed: np.ndarray | None = None) -> dict:
    """Reweight one shard to carry a non-zero invariant.

    Returns the achieved feature vector in the CELL measure, the per-event pair
    weights for refitting, and the diagnostics that decide whether the result
    may be used at all.
    """
    scores, velocity, omega, axis = particle_scores(run)
    if family not in scores:
        raise ValueError(f"no excitation score for {family!r}; "
                         f"have {sorted(scores)}")
    count = len(velocity) // 2
    logr = float(eta) * _normal_score(scores[family])
    logr -= logr.max()
    particle = np.exp(logr)

    # pair ratio is the product of the two one-particle ratios
    pair = particle[:count] * particle[count:]
    ess = float(pair.sum() ** 2 / np.sum(pair * pair)) / count
    share = float(pair.max() / pair.sum())

    # the achieved invariants must be read in the CELL measure: the attempts are
    # collision-flux weighted, and left uncorrected a Maxwellian shard reports
    # a2_tr = -0.0322 instead of zero
    if relative_speed is None:
        values = np.asarray(run.attempts["values"])
        relative_speed = np.linalg.norm(
            _vec(values, AI, "c1") - _vec(values, AI, "c2"), axis=1)
    debias = 1.0 / np.maximum(np.concatenate([relative_speed] * 2), 1.0e-30)
    weight = particle * debias
    weight = weight / weight.sum()
    positions = (np.random.default_rng(0x0CE11).random()
                 + np.arange(len(weight))) / len(weight)
    pick = np.searchsorted(np.cumsum(weight), positions, side="right").clip(
        0, len(weight) - 1)
    features, _ = cell_invariants(
        velocity[pick], omega[pick], axis[pick],
        float(run.metadata["mass"]), float(run.metadata["moi_perpendicular"]))
    return {
        "family": family, "eta": float(eta),
        "features": dict(zip(FEATURE_NAMES, features.tolist())),
        "pair_weight": pair,
        "ess_fraction": ess,
        "max_weight_share": share,
        "usable": bool(ess >= MINIMUM_ESS and share <= MAXIMUM_WEIGHT_SHARE
                       and family not in UNIDENTIFIABLE),
        "one_sided": family in ONE_SIDED,
        "unidentifiable": family in UNIDENTIFIABLE,
    }


def design(run, families=None, amplitudes=(-0.6, -0.3, 0.3, 0.6)) -> list[dict]:
    """One reweighted ensemble per (family, amplitude)."""
    scores, _, _, _ = particle_scores(run)
    if families is None:
        families = [f for f in scores if f not in UNIDENTIFIABLE]
    else:
        families = list(families)
    values = np.asarray(run.attempts["values"])
    speed = np.linalg.norm(_vec(values, AI, "c1") - _vec(values, AI, "c2"), axis=1)
    return [excite(run, family, eta, speed)
            for family in families for eta in amplitudes]
