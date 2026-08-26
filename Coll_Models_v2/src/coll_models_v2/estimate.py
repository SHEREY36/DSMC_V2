"""Block-bootstrap estimators for clock, total loss, routing, and VSS."""

from __future__ import annotations

import json

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES, load_run, validate_run
from dsmc_v2_contracts.io import OI, attempt_scores

from .vss import alpha_eff_from_b2, legendre


def _check_compatible(runs) -> None:
    reference = runs[0].metadata
    for key in ("alpha", "theta", "aspect_ratio", "velocity_scale", "omega_scale", "proposal_area"):
        values = np.array([float(run.metadata[key]) for run in runs])
        if not np.allclose(values, float(reference[key]), rtol=2.0e-12, atol=2.0e-12):
            raise ValueError(f"incompatible shard metadata for {key}: {values.tolist()}")
    for run in runs:
        qa = validate_run(run)
        if qa["status"] != "pass":
            raise ValueError(json.dumps(qa, indent=2))


def _sufficient_statistics(runs) -> np.ndarray:
    """Rows are bootstrap blocks; columns contain all estimator sums."""
    # ntry, nhit, sum_delta, sum_dtr, sum_e, sum_b2, then five 16-vectors.
    stats = np.zeros((128, 6 + 5 * 16), dtype=float)
    for run in runs:
        scores = attempt_scores(run)
        attempts, outcomes = np.asarray(run.attempts), np.asarray(run.outcomes)
        outcome_map = {(int(row["event_id"]), int(row["attempt_index"])): row
                       for row in outcomes}
        hit = attempts["hit"].astype(bool)
        delta = np.zeros(len(attempts))
        dtr = np.zeros(len(attempts))
        energy = np.zeros(len(attempts))
        b2 = np.zeros(len(attempts))
        for i in np.flatnonzero(hit):
            row = outcome_map[(int(attempts[i]["event_id"]), int(attempts[i]["attempt_index"]))]
            value = row["values"]
            delta[i], dtr[i], energy[i] = value[OI["delta_total"]], value[OI["delta_tr"]], value[OI["e_initial"]]
            cosine = sum(value[OI[f"ghat_pre_{a}"]] * value[OI[f"ghat_post_{a}"]] for a in "xyz")
            b2[i] = 1.0 - legendre(2, cosine)
        seed_shift = (int(run.metadata["seed"]) * 31) % 128
        blocks = (attempts["block_id"].astype(int) + seed_shift) % 128
        for block in range(128):
            mask, hmask = blocks == block, (blocks == block) & hit
            row = stats[block]
            row[:6] += (np.sum(mask), np.sum(hmask), np.sum(delta[mask]),
                        np.sum(dtr[mask]), np.sum(energy[mask]), np.sum(b2[mask]))
            row[6:22] += np.sum(scores[mask], axis=0)
            row[22:38] += np.sum(scores[hmask], axis=0)
            row[38:54] += np.sum(delta[mask, None] * scores[mask], axis=0)
            row[54:70] += np.sum(dtr[mask, None] * scores[mask], axis=0)
            row[70:86] += np.sum(energy[mask, None] * scores[mask], axis=0)
    return stats


def _evaluate(sums: np.ndarray, alpha: float, proposal_area: float,
              aspect_ratio: float) -> np.ndarray:
    ntry, nhit, sd, st, se, sb2 = sums[:6]
    if ntry <= 0 or nhit <= 0:
        raise ValueError("empty attempt or hit denominator")
    sigma = proposal_area * nhit / ntry
    p0 = nhit / ntry
    kappa = sums[22:38] / nhit - sums[6:22] / ntry
    eta_sigma = kappa / max(1.0 - p0, 1.0e-12)
    b2 = sb2 / nhit
    try:
        alpha_eff = alpha_eff_from_b2(b2)
    except ValueError:
        alpha_eff = np.nan
    if alpha >= 1.0:
        return np.concatenate(([sigma, np.nan, np.nan, b2, alpha_eff],
                               kappa, eta_sigma, np.full(32, np.nan)))
    if sd <= 0.0 or se <= 0.0 or st <= 0.0:
        raise ValueError("non-positive energy/loss/routing denominator")
    gamma, ftr = sd / se, st / sd
    sphere_routing = np.isclose(aspect_ratio, 1.0) and np.isclose(ftr, 1.0, atol=1.0e-10)
    if not (0.0 < gamma < 1.0 and (0.0 < ftr < 1.0 or sphere_routing)):
        raise ValueError(f"bounded means required; got Gamma={gamma}, Ftr={ftr}")
    ell_gamma = sums[38:54] / sd - sums[70:86] / se
    ell_ftr = sums[54:70] / st - sums[38:54] / sd
    eta_gamma = ell_gamma / (1.0 - gamma)
    eta_ftr = np.zeros(16) if sphere_routing else ell_ftr / (1.0 - ftr)
    return np.concatenate(([sigma, gamma, ftr, b2, alpha_eff],
                           kappa, eta_sigma, eta_gamma, eta_ftr))


def estimate_node(run_directories, n_bootstrap: int = 2000,
                  bootstrap_seed: int = 20260826) -> dict:
    runs = [load_run(path) for path in run_directories]
    if not runs:
        raise ValueError("at least one run directory is required")
    _check_compatible(runs)
    metadata = runs[0].metadata
    blocks = _sufficient_statistics(runs)
    estimate = _evaluate(np.sum(blocks, axis=0), float(metadata["alpha"]),
                         float(metadata["proposal_area"]), float(metadata["aspect_ratio"]))
    rng = np.random.default_rng(bootstrap_seed)
    boot = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, 128, size=128)
        try:
            boot.append(_evaluate(np.sum(blocks[chosen], axis=0), float(metadata["alpha"]),
                                  float(metadata["proposal_area"]), float(metadata["aspect_ratio"])))
        except ValueError:
            continue
    boot = np.asarray(boot)
    if len(boot) < max(20, n_bootstrap // 2):
        raise ValueError("too few valid bootstrap replicates")
    names = (["sigma0", "Gamma0", "Ftr0", "B2", "alpha_eff"]
             + [f"kappa_{name}" for name in FEATURE_NAMES]
             + [f"eta_sigma_{name}" for name in FEATURE_NAMES]
             + [f"eta_gamma_{name}" for name in FEATURE_NAMES]
             + [f"eta_ftr_{name}" for name in FEATURE_NAMES])
    rows = {}
    for i, name in enumerate(names):
        finite = boot[:, i][np.isfinite(boot[:, i])]
        rows[name] = {
            "estimate": float(estimate[i]) if np.isfinite(estimate[i]) else None,
            "standard_error": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else None,
            "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else None,
        }
    finite_columns = np.all(np.isfinite(boot), axis=0)
    covariance = np.full((len(names), len(names)), np.nan)
    if np.count_nonzero(finite_columns) > 1:
        covariance[np.ix_(finite_columns, finite_columns)] = np.cov(boot[:, finite_columns], rowvar=False)
    return {
        "schema_version": "2.0.0", "alpha": float(metadata["alpha"]),
        "theta": float(metadata["theta"]), "aspect_ratio": float(metadata["aspect_ratio"]),
        "n_attempts": int(sum(len(run.attempts) for run in runs)),
        "n_outcomes": int(sum(len(run.outcomes) for run in runs)),
        "bootstrap_replicates": int(len(boot)), "quantities": rows,
        "quantity_order": names, "covariance": covariance.tolist(),
        "qa": {"vss_representable": bool(np.isfinite(estimate[4]))},
    }
