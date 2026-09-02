"""Conversion between CTC hit-conditioned outcomes and the DSMC measure."""

from __future__ import annotations

import numpy as np

from dsmc_v2_contracts.io import AI, RunDataV2, _vec, attempt_scores


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


def outcome_attempt_indices(run: RunDataV2) -> np.ndarray:
    """Indices mapping keyed outcomes to their corresponding hit attempts."""
    lookup = {(int(row["event_id"]), int(row["attempt_index"])): i
              for i, row in enumerate(run.attempts)}
    try:
        return np.array([lookup[(int(row["event_id"]), int(row["attempt_index"]))]
                         for row in run.outcomes], dtype=int)
    except KeyError as exc:
        raise ValueError(f"outcome has no matching attempt: {exc.args[0]}") from exc


def outcome_weights(run: RunDataV2, normalise: bool = True) -> np.ndarray:
    """Inverse-area weights converting accepted outcomes to the DSMC measure."""
    area = projected_excluded_area(run)[outcome_attempt_indices(run)]
    weight = 1.0 / area
    if normalise:
        weight *= len(weight) / np.sum(weight)
    return weight


def effective_sample_size(weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=float)
    return float(np.sum(weight) ** 2 / np.sum(weight * weight))


def propensity_diagnostics(run: RunDataV2) -> dict:
    """Calibrate geometric hit propensity without comparing to the BL clock."""
    area = projected_excluded_area(run)
    proposal_area = float(run.metadata["proposal_area"])
    predicted = area / proposal_area
    hit = np.asarray(run.attempts["hit"], dtype=float)
    if np.any(predicted > 1.0 + 1.0e-10):
        raise ValueError("projected area exceeds the generator proposal area")
    residual = hit - predicted
    standard_error = float(np.std(residual, ddof=1) / np.sqrt(len(residual))) \
        if len(residual) > 1 else np.inf
    difference = float(np.mean(residual))
    return {
        "observed_hit_fraction": float(np.mean(hit)),
        "predicted_hit_fraction": float(np.mean(predicted)),
        "difference": difference,
        "standard_error": standard_error,
        "z_score": difference / standard_error if standard_error > 0.0 else np.inf,
        "pass": bool(abs(difference) <= 3.0 * standard_error),
        "maximum_predicted_propensity": float(np.max(predicted)),
    }


def proposal_balance_diagnostics(run: RunDataV2) -> dict:
    """Check that inverse-area hit weighting recovers all-proposal moments."""
    scores = attempt_scores(run)
    indices = outcome_attempt_indices(run)
    weight = outcome_weights(run)
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
