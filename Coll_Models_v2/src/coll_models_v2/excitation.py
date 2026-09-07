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

# No production invariant is structurally absent.  In particular W2 is the
# U-statistic |<omega>|^2, not axial spin <(omega.u)^2>.  Smooth rods impose
# omega.u=0 particle by particle, but a tangent spin population can still have
# a non-zero mean laboratory-frame omega and therefore a resolvable W2.
UNIDENTIFIABLE = ()

# Non-negative by construction: these are contractions, so both tilt directions
# raise them and only positive amplitudes carry information. That is the
# invariant's own property, not a limitation of the tilt.
ONE_SIDED = ("PiPi", "QQ", "RtRt", "qtr2", "qrot2", "W2")


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
    c = peculiar / float(run.metadata["velocity_scale"])
    w = omega / float(run.metadata["omega_scale"])
    axis = axis / np.linalg.norm(axis, axis=1)[:, None]
    speed2 = np.sum(c * c, axis=1)
    spin2 = np.sum(w * w, axis=1)

    # One representative component of each irreducible O(3) tensor/vector.
    # Rotational invariance makes the laboratory direction immaterial; using
    # components lets signed cross contractions be generated deliberately.
    pi = 2.0 * (c[:, 0] ** 2 - c[:, 1] ** 2)
    qq = 1.5 * (axis[:, 0] ** 2 - axis[:, 1] ** 2)
    rr = 3.0 * (w[:, 0] ** 2 - w[:, 1] ** 2)
    rt = rr + qq
    qtr = 0.8 * c[:, 0] * (speed2 - 2.5)
    qrot = 2.0 * c[:, 0] * (spin2 - 1.0)

    def coupled(left, right, sign=1.0):
        # Equalise the marginal score scales before forming parallel/opposed
        # perturbations. Otherwise the larger-variance field silently owns the
        # supposed cross-invariant excitation.
        return (_normal_score(left) + sign * _normal_score(right)) / np.sqrt(2.0)

    return {
        "a2_tr": speed2,
        "a2_rot": spin2,
        "a11": (speed2 - np.mean(speed2)) * (spin2 - np.mean(spin2)),
        "A_cu": np.einsum("ni,ni->n", c, axis) ** 2 - speed2 / 3.0,
        # stress: a component, not the contraction
        "PiPi": pi,
        "QQ": qq,
        "RtRt": rt,
        # Parallel and opposed component tilts are both required: changing the
        # sign of eta flips both fields and leaves their cross contraction's
        # sign unchanged.
        "PiQ": coupled(pi, qq),
        "PiQ_opposed": coupled(pi, qq, -1.0),
        "PiRt": coupled(pi, rt),
        "PiRt_opposed": coupled(pi, rt, -1.0),
        "QRt": coupled(qq, rt),
        "QRt_opposed": coupled(qq, rt, -1.0),
        # heat flux: an odd moment, so the sign of eta is meaningful
        "qtr2": qtr,
        "qrot2": qrot,
        "qtr_qrot": coupled(qtr, qrot),
        "qtr_qrot_opposed": coupled(qtr, qrot, -1.0),
        # Mean lab-frame spin, compatible with the smooth-rod tangent
        # constraint.  The former axial score (omega.u)^2 was identically zero.
        "W2": w[:, 0],
    }, velocity, omega, axis


def excite(run, family: str, eta: float,
           relative_speed: np.ndarray | None = None) -> dict:
    """Reweight one shard to carry a non-zero invariant.

    Returns the achieved feature vector in the CELL measure, the per-event pair
    weights for refitting, and the diagnostics that decide whether the result
    may be used at all.
    """
    result = excite_runs(
        [run], family, eta,
        None if relative_speed is None else [relative_speed])
    result["pair_weight"] = result.pop("attempt_weights")[0]
    return result


def excite_runs(runs, family: str, eta: float,
                relative_speeds: list[np.ndarray] | None = None) -> dict:
    """Construct one importance-sampled excitation across one or more shards.

    Ranking is global across the shards, and the returned attempt weights stay
    shard-local so accepted outcomes can be joined through their recorded
    attempt indices.  This is the executable Radon--Nikodym bridge from the
    baseline CTC ensemble to the deliberately perturbed incoming gas.
    """
    runs = list(runs)
    if not runs:
        raise ValueError("at least one baseline run is required")
    parts = [particle_scores(run) for run in runs]
    available = set(parts[0][0])
    if family not in available or any(set(part[0]) != available for part in parts):
        raise ValueError(f"no common excitation score for {family!r}; "
                         f"have {sorted(available)}")
    masses = {float(run.metadata["mass"]) for run in runs}
    inertias = {float(run.metadata["moi_perpendicular"]) for run in runs}
    if len(masses) != 1 or len(inertias) != 1:
        raise ValueError("excitation shards do not share particle properties")

    lengths = [len(part[1]) for part in parts]
    score = _normal_score(np.concatenate([part[0][family] for part in parts]))
    logr = float(eta) * score
    logr -= logr.max()
    particle_all = np.exp(logr)
    particle_parts, start = [], 0
    for length in lengths:
        particle_parts.append(particle_all[start:start + length])
        start += length

    attempt_weights = []
    for particle in particle_parts:
        count = len(particle) // 2
        attempt_weights.append(particle[:count] * particle[count:])
    pair_all = np.concatenate(attempt_weights)
    ess = float(pair_all.sum() ** 2 / np.sum(pair_all * pair_all)) / len(pair_all)
    share = float(pair_all.max() / pair_all.sum())

    # The achieved invariants must be read in the CELL measure: the attempts are
    # collision-flux weighted, and left uncorrected a Maxwellian shard reports
    # a2_tr = -0.0322 instead of zero.
    if relative_speeds is None:
        relative_speeds = []
        for run in runs:
            values = np.asarray(run.attempts["values"])
            relative_speeds.append(np.linalg.norm(
                _vec(values, AI, "c1") - _vec(values, AI, "c2"), axis=1))
    if len(relative_speeds) != len(runs):
        raise ValueError("one relative-speed array is required per shard")
    debias = np.concatenate([
        1.0 / np.maximum(np.concatenate([np.asarray(speed, dtype=float)] * 2),
                         1.0e-30)
        for speed in relative_speeds
    ])
    weight = particle_all * debias
    weight = weight / weight.sum()
    positions = (np.random.default_rng(0x0CE11).random()
                 + np.arange(len(weight))) / len(weight)
    pick = np.searchsorted(np.cumsum(weight), positions, side="right").clip(
        0, len(weight) - 1)
    velocity = np.concatenate([part[1] for part in parts])
    omega = np.concatenate([part[2] for part in parts])
    axis = np.concatenate([part[3] for part in parts])
    features, _ = cell_invariants(
        velocity[pick], omega[pick], axis[pick], masses.pop(), inertias.pop())
    target = family.removesuffix("_opposed")
    return {
        "family": family, "eta": float(eta),
        "target_feature": target,
        "features": dict(zip(FEATURE_NAMES, features.tolist())),
        "attempt_weights": attempt_weights,
        "ess_fraction": ess,
        "max_weight_share": share,
        "usable": bool(ess >= MINIMUM_ESS and share <= MAXIMUM_WEIGHT_SHARE
                       and target not in UNIDENTIFIABLE),
        "one_sided": target in ONE_SIDED,
        "unidentifiable": target in UNIDENTIFIABLE,
    }


def estimate_excitation(run_directories, family: str, eta: float,
                        ensemble_id: int, anchor: tuple,
                        n_bootstrap: int = 200,
                        propensity_offsets: int | None = 128) -> dict:
    """Fit a virtual excited node directly from baseline CTC shards."""
    from .estimate import estimate_node
    from .pipeline import precision_status

    paths = list(run_directories)
    runs = [load_run(path) for path in paths]
    excited = excite_runs(runs, family, eta)
    if not excited["usable"]:
        raise ValueError(
            f"excitation {family} eta={eta} fails overlap: "
            f"ESS={excited['ess_fraction']:.3f}, "
            f"max_share={excited['max_weight_share']:.3e}")
    metadata = {key: value for key, value in excited.items()
                if key not in ("attempt_weights", "features")}
    result = estimate_node(
        paths, n_bootstrap=n_bootstrap,
        propensity_offsets=propensity_offsets, anchor=anchor,
        attempt_weights=excited["attempt_weights"],
        cell_features_override=np.array([
            excited["features"][name] for name in FEATURE_NAMES]),
        ensemble_id_override=int(ensemble_id), excitation=metadata)
    passed, reasons = precision_status(result)
    result["qa"].update(precision_pass=passed, continuation_reasons=reasons)
    return result


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
