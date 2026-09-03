"""Conversion between CTC hit-conditioned outcomes and the DSMC measure.

The generator accepts a proposal when the two spherocylinders touch anywhere
along the encounter, and they turn while they close, so its acceptance
probability is the *dynamic* excluded area, not the static shadow ``A_perp``.
The two differ by 9 to 33 percent at aspect ratios 2 and 3, growing as theta
falls, and the difference is a clean monotone function of how far the rods
rotate over one interaction length.

That is physics, not a staging artefact: contact is only possible while the
centre separation is below ``L + D``, the generator starts beyond that, and free
rotation maps an isotropic director distribution to an isotropic one, so the
answer cannot depend on the staging distance -- and measurably does not.

The acceptance is nevertheless pure kinematics: no forces act before contact.
``kinematic_propensity`` therefore computes it directly from the stored
pre-collision state, and ``1 / propensity`` is the exact Radon-Nikodym weight
onto the DSMC's orientation-blind collision measure.
"""

from __future__ import annotations

import numpy as np

from dsmc_v2_contracts.io import (AI, OI, RunDataV2, _vec, attempt_energy,
                                  attempt_scores)

from .projections import _legendre_nodes, incoming_partition_density


def projected_excluded_area(run: RunDataV2) -> np.ndarray:
    """Projected excluded area for every incoming CTC proposal.

    The formula is evaluated from the stored pre-collision directors and
    relative-velocity direction, so schema-2.1 shards need no rewrite.
    """
    values = np.asarray(run.attempts["values"])
    c1, c2 = _vec(values, AI, "c1"), _vec(values, AI, "c2")
    u1, u2 = _vec(values, AI, "u1"), _vec(values, AI, "u2")
    g = c1 - c2
    gnorm = np.linalg.norm(g, axis=1)
    if np.any(gnorm <= 1.0e-30):
        raise ValueError("zero relative speed in CTC proposals")
    ghat = g / gnorm[:, None]
    diameter = float(run.metadata.get("diameter", 1.0))
    length = (float(run.metadata["aspect_ratio"]) - 1.0) * diameter
    s1 = np.linalg.norm(np.cross(u1, ghat), axis=1)
    s2 = np.linalg.norm(np.cross(u2, ghat), axis=1)
    triple = np.abs(np.einsum("ni,ni->n", ghat, np.cross(u1, u2)))
    area = (np.pi * diameter * diameter
            + 2.0 * diameter * length * (s1 + s2)
            + length * length * triple)
    if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("projected excluded area must be finite and positive")
    return area


DEFAULT_OFFSETS = 128
RELATIVE_BIAS_TOLERANCE = 0.02
# Measured convergence at the worst node (AR = 3, theta = 0.2, median turn 7.8
# radians): the encounter integral settles by ~192 steps and is already within
# 0.1 percent at 96, so 24 steps per radian of turn is ample.
STEPS_PER_RADIAN = 24
MINIMUM_STEPS = 64
MAXIMUM_STEPS = 768
STAGING_FACTOR = 1.01      # HS_CTC_v2 initialize.f90: BMAX = 1.01 (L + D)


def _segment_distance_squared(separation: np.ndarray, first: np.ndarray,
                              second: np.ndarray, half_length: float) -> np.ndarray:
    """Squared distance between the segments 0 +- hL*first and r +- hL*second."""
    dot = np.einsum("ni,ni->n", first, second)
    project_first = np.einsum("ni,ni->n", first, separation)
    project_second = np.einsum("ni,ni->n", second, separation)
    determinant = 1.0 - dot * dot
    regular = determinant > 1.0e-9
    safe = np.where(regular, determinant, 1.0)
    s = np.where(regular, (project_first - dot * project_second) / safe, 0.0)
    t = np.where(regular, (dot * project_first - project_second) / safe, -project_second)
    s = np.clip(s, -half_length, half_length)
    t = np.clip(t, -half_length, half_length)
    for _ in range(3):          # alternating projection; converges in a few passes
        t = np.clip(dot * s - project_second, -half_length, half_length)
        s = np.clip(dot * t + project_first, -half_length, half_length)
    gap = separation + t[:, None] * second - s[:, None] * first
    return np.einsum("ni,ni->n", gap, gap)


def _disc_lattice(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Sunflower lattice on the unit disc: low discrepancy for a smooth region."""
    index = np.arange(count) + 0.5
    radius = np.sqrt(index / count)
    angle = index * np.pi * (3.0 - np.sqrt(5.0))
    return radius * np.cos(angle), radius * np.sin(angle)


def _transverse_basis(ghat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.tile(np.array([1.0, 0.0, 0.0]), (len(ghat), 1))
    trial[np.abs(ghat[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    first = np.cross(ghat, trial)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    return first, np.cross(ghat, first)


def encounter_propensity(ghat: np.ndarray, speed: np.ndarray,
                         director: list[np.ndarray], spin_vector: list[np.ndarray],
                         diameter: float, length: float, staging: float,
                         offsets: int = DEFAULT_OFFSETS, steps: int | None = None,
                         seed: int = 20260902, block: int = 4194304) -> np.ndarray:
    """Acceptance probability of a force-free spherocylinder encounter.

    The pair starts at longitudinal separation ``staging`` with a transverse
    offset drawn uniformly over the generator's impact box, closes at constant
    ``speed`` along ``ghat`` while both rods turn freely, and is accepted if the
    axes ever come within ``diameter``.  Returns the probability per proposal.
    """
    half_length, reach = 0.5 * length, length + diameter
    if staging < reach:
        raise ValueError("staging distance must exceed the contact reach L + D")
    count = len(speed)
    if np.any(speed <= 1.0e-30):
        raise ValueError("zero relative speed in CTC proposals")
    basis_one, basis_two = _transverse_basis(ghat)

    spin = [np.linalg.norm(w, axis=1) for w in spin_vector]
    spin_axis = [w / np.maximum(m, 1.0e-300)[:, None]
                 for w, m in zip(spin_vector, spin)]

    # The rotation number is heavy tailed -- a median near 8 radians with a tail
    # past 100 -- so a handful of fast rotators would set the step count for
    # every event. Order the work by rotation number and close each block when
    # its cost, events times the resolution that block needs, hits a budget.
    turn = (spin[0] + spin[1]) * reach / speed
    order = np.arange(count) if steps is not None else np.argsort(turn)
    per_event_steps = (np.full(count, int(steps)) if steps is not None else
                       np.clip(np.ceil(STEPS_PER_RADIAN * turn),
                               MINIMUM_STEPS, MAXIMUM_STEPS).astype(int))

    phase = np.random.default_rng(seed).random(count) * 2.0 * np.pi
    lattice_x, lattice_y = _disc_lattice(offsets)
    struck = np.empty(count)
    budget = max(block, MINIMUM_STEPS * offsets)

    start = 0
    while start < count:
        span = 1
        while start + span < count:
            trial = span + 1
            if trial * offsets * int(per_event_steps[order[start + trial - 1]]) > budget:
                break
            span = trial
        stop = start + span
        cut = order[start:stop]
        block_steps = int(np.max(per_event_steps[cut]))
        cosine = np.cos(phase[cut])[:, None]
        sine = np.sin(phase[cut])[:, None]
        local_x = reach * (lattice_x[None, :] * cosine - lattice_y[None, :] * sine)
        local_y = reach * (lattice_x[None, :] * sine + lattice_y[None, :] * cosine)
        impact = (local_x[:, :, None] * basis_one[cut, None, :]
                  + local_y[:, :, None] * basis_two[cut, None, :]).reshape(-1, 3)
        window = np.sqrt(np.maximum(
            reach * reach - (local_x * local_x + local_y * local_y).ravel(), 0.0))
        pace = np.repeat(speed[cut], offsets)
        along = np.repeat(ghat[cut], offsets, axis=0)
        entry, departure = (staging - window) / pace, (staging + window) / pace
        axis = [np.repeat(a[cut], offsets, axis=0) for a in spin_axis]
        rate = [np.repeat(m[cut], offsets) for m in spin]
        start_direction = [np.repeat(u[cut], offsets, axis=0) for u in director]
        missing = np.ones(len(pace), dtype=bool)
        for step in range(block_steps + 1):
            moment = entry + (departure - entry) * step / block_steps
            separation = impact + (staging - pace * moment)[:, None] * along
            turned = []
            for i in (0, 1):
                angle = (rate[i] * moment)[:, None]
                base = start_direction[i]
                turned.append(base * np.cos(angle)
                              + np.cross(axis[i], base) * np.sin(angle)
                              + axis[i] * np.einsum("ni,ni->n", axis[i], base)[:, None]
                              * (1.0 - np.cos(angle)))
            missing &= _segment_distance_squared(
                separation, turned[0], turned[1], half_length) >= diameter * diameter
            if not missing.any():
                break
        struck[cut] = (~missing).reshape(stop - start, offsets).mean(axis=1)
        start = stop

    # The lattice covers the disc of radius L+D; nothing outside it can touch.
    return struck * (np.pi * reach * reach) / (4.0 * staging * staging)


def kinematic_propensity(run: RunDataV2, offsets: int = DEFAULT_OFFSETS,
                         steps: int | None = None, seed: int = 20260902,
                         block: int = 4194304,
                         indices: np.ndarray | None = None) -> np.ndarray:
    """Probability that a stored proposal is accepted by the generator.

    Integrates the force-free encounter over the impact-parameter plane. The
    lattice is rotated by a per-event random phase, so the estimator is unbiased
    while keeping the low discrepancy of a deterministic lattice.

    The generator's own transverse basis is not recorded, but the effective area
    is invariant under rotation about ``ghat``, so any orthonormal basis will do.
    """
    values = np.asarray(run.attempts["values"])
    if indices is not None:
        values = values[np.asarray(indices, dtype=int)]
    diameter = float(run.metadata.get("diameter", 1.0))
    length = (float(run.metadata["aspect_ratio"]) - 1.0) * diameter
    relative = _vec(values, AI, "c1") - _vec(values, AI, "c2")
    speed = np.linalg.norm(relative, axis=1)
    if np.any(speed <= 1.0e-30):
        raise ValueError("zero relative speed in CTC proposals")
    return encounter_propensity(
        relative / speed[:, None], speed,
        [_vec(values, AI, "u1"), _vec(values, AI, "u2")],
        [_vec(values, AI, "omega1"), _vec(values, AI, "omega2")],
        diameter, length, STAGING_FACTOR * (length + diameter),
        offsets=offsets, steps=steps, seed=seed, block=block)


def debiased_inverse(propensity: np.ndarray, offsets: int) -> np.ndarray:
    """1/p with the leading Monte-Carlo bias of the plug-in estimator removed.

    ``E[1/p_hat] = (1/p)(1 + (1-p)/(M p)) + O(M^-2)``, and the inflation runs
    from 0.1 percent at p = 0.8 to 5 percent at p = 0.1, so it does not cancel
    in the normalised weights.
    """
    propensity = np.asarray(propensity, dtype=float)
    if np.any(propensity <= 0.0):
        raise ValueError("propensity must be positive; increase the offset count")
    inflation = (1.0 - propensity) / (offsets * propensity)
    return (1.0 / propensity) / (1.0 + inflation)


def outcome_attempt_indices(run: RunDataV2) -> np.ndarray:
    """Indices mapping keyed outcomes to their corresponding hit attempts."""
    lookup = {(int(row["event_id"]), int(row["attempt_index"])): i
              for i, row in enumerate(run.attempts)}
    try:
        return np.array([lookup[(int(row["event_id"]), int(row["attempt_index"]))]
                         for row in run.outcomes], dtype=int)
    except KeyError as exc:
        raise ValueError(f"outcome has no matching attempt: {exc.args[0]}") from exc


def outcome_weights(run: RunDataV2, normalise: bool = True,
                    propensity: np.ndarray | None = None,
                    offsets: int = DEFAULT_OFFSETS) -> np.ndarray:
    """Weights converting accepted outcomes onto the DSMC collision measure.

    With ``propensity`` supplied the weight is the exact Radon-Nikodym
    derivative ``1 / P(accept | state)``. Without it the caller gets the old
    static-shadow weight ``1 / A_perp``, which is retained only for A/B runs:
    it misses the rotational enhancement and biases the accepted ensemble
    towards fast-spinning pairs.
    """
    index = outcome_attempt_indices(run)
    if propensity is None:
        weight = 1.0 / projected_excluded_area(run)[index]
    else:
        weight = debiased_inverse(np.asarray(propensity, dtype=float)[index], offsets)
    if normalise:
        weight *= len(weight) / np.sum(weight)
    return weight


def effective_sample_size(weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=float)
    return float(np.sum(weight) ** 2 / np.sum(weight * weight))


def propensity_diagnostics(run: RunDataV2, propensity: np.ndarray | None = None,
                           bins: int = 6) -> dict:
    """Does the predicted acceptance reproduce the acceptance actually observed?

    This stays a genuine test rather than a tautology because ``propensity`` is
    an independent physical calculation, not a model fitted to the hit flags.
    Alongside the z-score, which necessarily tightens as the event count grows,
    the relative bias is reported so the tolerance can be stated as a physical
    accuracy rather than as a significance level.
    """
    hit = np.asarray(run.attempts["hit"], dtype=float)
    if propensity is None:
        predicted = projected_excluded_area(run) / float(run.metadata["proposal_area"])
        model = "static_projected_area"
        if np.any(predicted > 1.0 + 1.0e-10):
            raise ValueError("projected area exceeds the generator proposal area")
    else:
        predicted = np.asarray(propensity, dtype=float)
        model = "kinematic_encounter"
    residual = hit - predicted
    standard_error = float(np.std(residual, ddof=1) / np.sqrt(len(residual))) \
        if len(residual) > 1 else np.inf
    difference = float(np.mean(residual))
    edges = np.quantile(predicted, np.linspace(0.0, 1.0, bins + 1))
    edges[-1] += 1.0e-12
    which = np.clip(np.searchsorted(edges, predicted, side="right") - 1, 0, bins - 1)
    calibration = [{
        "predicted": float(np.mean(predicted[which == b])),
        "observed": float(np.mean(hit[which == b])),
        "count": int(np.count_nonzero(which == b)),
    } for b in range(bins) if np.any(which == b)]
    observed = float(np.mean(hit))
    return {
        "model": model,
        "observed_hit_fraction": observed,
        "predicted_hit_fraction": float(np.mean(predicted)),
        "difference": difference,
        "relative_bias": difference / observed if observed > 0.0 else np.inf,
        "standard_error": standard_error,
        "z_score": difference / standard_error if standard_error > 0.0 else np.inf,
        "pass": bool(abs(difference) <= 3.0 * standard_error
                     or abs(difference) <= RELATIVE_BIAS_TOLERANCE * observed),
        "maximum_predicted_propensity": float(np.max(predicted)),
        "calibration": calibration,
    }


def incoming_partition_diagnostics(run: RunDataV2, propensity: np.ndarray | None = None,
                                   offsets: int = DEFAULT_OFFSETS,
                                   quadrature: int = 256) -> dict:
    """Compare the sampled pre-collision partition against its exact law.

    Both modal energies are Gamma(2) under the generator -- the relative
    translational energy because the relative speed is drawn from the
    collision-weighted law, the rotational because four independent Maxwellian
    spin components make a chi-square with four degrees of freedom -- so

        p(z) proportional to z (1 - z) (z / theta + 1 - z)^-4.

    Note this is *not* ``theta / (theta + 1)``, which is the ratio of the means
    rather than the mean of the ratio: at theta = 0.2 they differ by 28 percent,
    so testing against it would report a healthy sampler as broken.

    Two comparisons are returned. Over all proposals it tests the generator.
    Over the accepted outcomes, reweighted by the inverse propensity, it tests
    the measure conversion, since acceptance is correlated with the relative
    speed and hence with the partition.
    """
    theta = float(run.metadata["theta"])
    values = np.asarray(run.attempts["values"])
    c1, c2 = _vec(values, AI, "c1"), _vec(values, AI, "c2")
    centre = 0.5 * (c1 + c2)
    mass = float(run.metadata["mass"])
    translational = mass * (np.sum((c1 - centre) ** 2, axis=1)
                            + np.sum((c2 - centre) ** 2, axis=1))
    partition = translational / attempt_energy(run)

    nodes, quad = _legendre_nodes(quadrature, 0.0, 1.0)
    mass_law = incoming_partition_density(theta, nodes) * quad
    mass_law /= np.sum(mass_law)
    expected = float(mass_law @ nodes)
    expected_spread = float(np.sqrt(mass_law @ (nodes - expected) ** 2))

    index = outcome_attempt_indices(run)
    weight = outcome_weights(run, propensity=propensity, offsets=offsets)
    accepted = float(np.sum(weight * partition[index]) / np.sum(weight))

    count = len(partition)
    standard_error = float(np.std(partition, ddof=1) / np.sqrt(count))
    difference = float(np.mean(partition) - expected)
    return {
        "theta": theta,
        "expected_mean": expected,
        "expected_spread": expected_spread,
        "proposal_mean": float(np.mean(partition)),
        "proposal_spread": float(np.std(partition, ddof=1)),
        "reweighted_accepted_mean": accepted,
        "difference": difference,
        "relative_difference": difference / expected,
        "standard_error": standard_error,
        "z_score": difference / standard_error if standard_error > 0.0 else np.inf,
        "pass": bool(abs(difference) <= 3.0 * standard_error
                     or abs(difference) <= RELATIVE_BIAS_TOLERANCE * expected),
    }


def proposal_balance_diagnostics(run: RunDataV2, propensity: np.ndarray | None = None,
                                 offsets: int = DEFAULT_OFFSETS) -> dict:
    """Check that inverse-propensity hit weighting recovers all-proposal moments.

    This is the test that survives once the propensity is estimated rather than
    assumed: reweighting undoes the acceptance bias only if the propensity is
    right, and nothing forces it to hold.
    """
    scores = attempt_scores(run)
    indices = outcome_attempt_indices(run)
    weight = outcome_weights(run, propensity=propensity, offsets=offsets)
    proposal_mean = np.mean(scores, axis=0)
    outcome_mean = np.sum(weight[:, None] * scores[indices], axis=0) / np.sum(weight)
    proposal_se = np.std(scores, axis=0, ddof=1) / np.sqrt(len(scores))
    ess = effective_sample_size(weight)
    centered = scores[indices] - outcome_mean
    outcome_variance = np.sum(weight[:, None] * centered * centered, axis=0) / np.sum(weight)
    outcome_se = np.sqrt(outcome_variance / ess)
    combined = np.hypot(proposal_se, outcome_se)
    z_score = (outcome_mean - proposal_mean) / np.maximum(combined, 1.0e-30)
    return {
        "proposal_mean": proposal_mean.tolist(),
        "inverse_area_outcome_mean": outcome_mean.tolist(),
        "z_score": z_score.tolist(),
        "maximum_absolute_z_score": float(np.max(np.abs(z_score))),
        "pass": bool(np.all(np.abs(z_score) <= 3.0)),
    }
