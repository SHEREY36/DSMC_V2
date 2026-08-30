"""Standalone CTC estimators for conservative v2.1 routing and scattering."""

from __future__ import annotations

import json

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES, load_run, validate_run
from dsmc_v2_contracts.io import OI, attempt_energy, attempt_scores

from .legacy_bl import LegacyBL
from .vss import alpha_eff_from_b2, legendre


N_BLOCKS = 128
N_SCALARS = 10


def frozen_cross_section(aspect_ratio: float, diameter: float = 1.0) -> float:
    """The validated v1 collision-clock cross-section; never fitted here."""
    ar = float(aspect_ratio)
    return float(np.pi * diameter * diameter * (0.32 * ar * ar + 0.694 * ar - 0.0213))


def _check_compatible(runs) -> None:
    reference = runs[0].metadata
    for key in ("alpha", "theta", "aspect_ratio", "velocity_scale", "omega_scale",
                "proposal_area", "mass", "moi_perpendicular"):
        values = np.array([float(run.metadata[key]) for run in runs])
        if not np.allclose(values, float(reference[key]), rtol=2.0e-12, atol=2.0e-12):
            raise ValueError(f"incompatible shard metadata for {key}: {values.tolist()}")
    for run in runs:
        qa = validate_run(run)
        if qa["status"] != "pass":
            raise ValueError(json.dumps(qa, indent=2))


def _sufficient_statistics(runs) -> np.ndarray:
    """Attempt-block rows used for bootstrap and shard-order invariant merging.

    Scalars are ntry, nhit, sum(delta), sum(delta_tr), sum(E_try), sum(B2),
    and sums of P1..P4 over hits.  Four 16-vectors then hold try scores,
    delta-weighted scores, delta_tr-weighted scores, and E_try-weighted scores.
    """
    stats = np.zeros((N_BLOCKS, N_SCALARS + 4 * len(FEATURE_NAMES)), dtype=float)
    for run in runs:
        scores = attempt_scores(run)
        e_try = attempt_energy(run)
        attempts, outcomes = np.asarray(run.attempts), np.asarray(run.outcomes)
        outcome_map = {(int(row["event_id"]), int(row["attempt_index"])): row
                       for row in outcomes}
        hit = attempts["hit"].astype(bool)
        delta = np.zeros(len(attempts))
        dtr = np.zeros(len(attempts))
        angular = np.zeros((len(attempts), 5))
        for i in np.flatnonzero(hit):
            row = outcome_map[(int(attempts[i]["event_id"]),
                               int(attempts[i]["attempt_index"]))]
            value = row["values"]
            delta[i] = value[OI["delta_total"]]
            dtr[i] = value[OI["delta_tr"]]
            cosine = float(np.clip(sum(
                value[OI[f"ghat_pre_{axis}"]] * value[OI[f"ghat_post_{axis}"]]
                for axis in "xyz"), -1.0, 1.0))
            angular[i, 0] = 1.0 - legendre(2, cosine)
            angular[i, 1:] = [legendre(order, cosine) for order in range(1, 5)]
        # A shard-specific permutation retains deterministic blocks while
        # preventing identically numbered blocks in every shard being coupled.
        shift = (int(run.metadata["seed"]) * 31) % N_BLOCKS
        blocks = (attempts["block_id"].astype(int) + shift) % N_BLOCKS
        for block in range(N_BLOCKS):
            mask = blocks == block
            hmask = mask & hit
            row = stats[block]
            row[:N_SCALARS] += (
                np.sum(mask), np.sum(hmask), np.sum(delta[mask]), np.sum(dtr[mask]),
                np.sum(e_try[mask]), np.sum(angular[mask, 0]),
                np.sum(angular[mask, 1]), np.sum(angular[mask, 2]),
                np.sum(angular[mask, 3]), np.sum(angular[mask, 4]),
            )
            start = N_SCALARS
            row[start:start + 16] += np.sum(scores[mask], axis=0)
            row[start + 16:start + 32] += np.sum(delta[mask, None] * scores[mask], axis=0)
            row[start + 32:start + 48] += np.sum(dtr[mask, None] * scores[mask], axis=0)
            row[start + 48:start + 64] += np.sum(e_try[mask, None] * scores[mask], axis=0)
    return stats


def _evaluate(sums: np.ndarray, metadata: dict, bl: LegacyBL) -> np.ndarray:
    ntry, nhit, sd, st, se, sb2, sp1, sp2, sp3, sp4 = sums[:N_SCALARS]
    if ntry <= 0 or nhit <= 0:
        raise ValueError("empty attempt or hit denominator")
    alpha = float(metadata["alpha"])
    theta = float(metadata["theta"])
    ar = float(metadata["aspect_ratio"])
    area = float(metadata["proposal_area"])
    sigma_poly = float(metadata.get("collision_cross_section", frozen_cross_section(ar)))
    sigma_ctc = area * nhit / ntry
    b2 = sb2 / nhit
    try:
        alpha_eff = alpha_eff_from_b2(b2)
    except ValueError:
        alpha_eff = np.nan
    scalar = [sigma_ctc, sigma_poly, (sigma_ctc - sigma_poly) / sigma_poly]
    if alpha >= 1.0:
        scalar.extend([np.nan] * 4)
    else:
        params = bl.parameters(alpha, ar)
        mean_gamma = params["mean_loss_fraction"]
        sbl = mean_gamma * se
        if min(sd, st, se, sbl) <= 0.0:
            raise ValueError("non-positive loss or routing denominator")
        f0 = area * st / (sigma_poly * sbl)
        fc = st / sd
        # These are modal production ratios, not probabilities.  Values above
        # one are valid when translation loses energy while rotation gains
        # part of that energy, so that delta_tr > delta_total.  The preserved
        # v1 kernel supports this through its unbounded f_tr and reservoir
        # handling.  Only a non-positive reference would invalidate the
        # multiplicative first-order closure used on the present design grid.
        if f0 <= 0.0 or fc <= 0.0:
            raise ValueError(f"routing reference must be positive; F0={f0}, FC={fc}")
        # C_M belongs to the production modal-routing closure. The BL-matched
        # F0 remains an audit quantity, while F_C is the reference consumed by
        # DSMC when its validated total-loss law is preserved.
        cm = fc / (3.0 * theta / (3.0 * theta + 2.0))
        total_audit = area * sd / (sigma_poly * sbl)
        scalar.extend([f0, cm, fc, total_audit])
    scalar.extend([b2, alpha_eff, sp1 / nhit, sp2 / nhit, sp3 / nhit, sp4 / nhit])

    try_scores = sums[N_SCALARS:N_SCALARS + 16]
    delta_scores = sums[N_SCALARS + 16:N_SCALARS + 32]
    dtr_scores = sums[N_SCALARS + 32:N_SCALARS + 48]
    energy_scores = sums[N_SCALARS + 48:N_SCALARS + 64]
    if alpha >= 1.0:
        beta = beta_ctc = np.full(16, np.nan)
    else:
        # Multiplication by the constant BL mean loss cancels, but retaining
        # the expression documents which preserved DSMC production is used.
        lbl = energy_scores / se
        beta = dtr_scores / st - lbl
        beta_ctc = dtr_scores / st - delta_scores / sd
    return np.concatenate((np.asarray(scalar), beta, beta_ctc, try_scores / ntry))


def _quantity_names() -> list[str]:
    return (["sigma_ctc", "sigma_polynomial", "sigma_relative_error",
             "F0", "C_M", "F_C", "total_loss_compatibility_ratio",
             "B2", "alpha_eff", "mean_P1", "mean_P2", "mean_P3", "mean_P4"]
            + [f"beta_{name}" for name in FEATURE_NAMES]
            + [f"beta_ctc_{name}" for name in FEATURE_NAMES]
            + [f"mean_try_score_{name}" for name in FEATURE_NAMES])


def estimate_node(run_directories, bl: LegacyBL, n_bootstrap: int = 2000,
                  bootstrap_seed: int = 20260826) -> dict:
    runs = [load_run(path) for path in run_directories]
    if not runs:
        raise ValueError("at least one run directory is required")
    _check_compatible(runs)
    metadata = runs[0].metadata
    shard_blocks = [_sufficient_statistics([run]) for run in runs]
    blocks = np.sum(shard_blocks, axis=0)
    estimate = _evaluate(np.sum(blocks, axis=0), metadata, bl)
    rng = np.random.default_rng(bootstrap_seed)
    boot = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, N_BLOCKS, size=N_BLOCKS)
        try:
            boot.append(_evaluate(np.sum(blocks[chosen], axis=0), metadata, bl))
        except ValueError:
            continue
    boot = np.asarray(boot)
    if len(boot) < max(20, n_bootstrap // 2):
        raise ValueError("too few valid bootstrap replicates")
    names = _quantity_names()
    quantities = {}
    for i, name in enumerate(names):
        finite = boot[:, i][np.isfinite(boot[:, i])]
        quantities[name] = {
            "estimate": float(estimate[i]) if np.isfinite(estimate[i]) else None,
            "standard_error": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else None,
            "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else None,
        }
    finite_columns = np.all(np.isfinite(boot), axis=0)
    covariance = np.full((len(names), len(names)), np.nan)
    if np.count_nonzero(finite_columns) > 1:
        covariance[np.ix_(finite_columns, finite_columns)] = np.cov(
            boot[:, finite_columns], rowvar=False)
    sigma = quantities["sigma_relative_error"]
    cross_section_pass = abs(sigma["estimate"]) <= 0.02 or (
        sigma["ci_low"] <= 0.0 <= sigma["ci_high"])
    loss = quantities["total_loss_compatibility_ratio"]
    loss_pass = float(metadata["alpha"]) >= 1.0 or (
        loss["ci_low"] is not None and loss["ci_low"] <= 1.10
        and loss["ci_high"] >= 0.90 and abs(loss["estimate"] - 1.0) <= 0.10)
    # A single block carrying too much of a weighted sum signals score-tail
    # instability; contributions are diagnosed, never clipped.
    total_dtr = np.sum(blocks[:, 3])
    loss_tail_fraction = float(np.max(np.abs(blocks[:, 3])) /
                               max(abs(total_dtr), 1.0e-300))
    weighted_score = blocks[:, N_SCALARS + 32:N_SCALARS + 48]
    score_tail_fraction = float(np.max(np.abs(weighted_score) /
        np.maximum(np.sum(np.abs(weighted_score), axis=0), 1.0e-300)))
    leave_one_out = []
    if len(shard_blocks) > 1:
        total = np.sum(shard_blocks, axis=0)
        for index, shard in enumerate(shard_blocks):
            value = _evaluate(np.sum(total - shard, axis=0), metadata, bl)
            leave_one_out.append({
                "omitted_shard": index, "F0": None if not np.isfinite(value[3]) else float(value[3]),
                "B2": float(value[7]),
                "beta": [None if not np.isfinite(x) else float(x) for x in value[13:29]],
            })
    return {
        "schema_version": "2.1.0", "alpha": float(metadata["alpha"]),
        "theta": float(metadata["theta"]), "aspect_ratio": float(metadata["aspect_ratio"]),
        "n_attempts": int(sum(len(run.attempts) for run in runs)),
        "n_outcomes": int(sum(len(run.outcomes) for run in runs)),
        "bl_parameters": bl.parameters(float(metadata["alpha"]),
                                         float(metadata["aspect_ratio"]))
        if float(metadata["alpha"]) < 1.0 else None,
        "bootstrap_replicates": int(len(boot)), "quantities": quantities,
        "quantity_order": names, "covariance": covariance.tolist(),
        "leave_one_shard_out": leave_one_out,
        "qa": {
            "cross_section_pass": bool(cross_section_pass),
            "total_loss_compatibility_pass": bool(loss_pass),
            "vss_representable": bool(np.isfinite(estimate[8])),
            "maximum_block_translational_loss_fraction": loss_tail_fraction,
            "maximum_block_weighted_score_fraction": score_tail_fraction,
            "score_tail_pass": loss_tail_fraction < 0.10 and score_tail_fraction < 0.25,
        },
    }
