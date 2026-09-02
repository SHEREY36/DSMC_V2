"""Node-wise estimation for the BL-compatible variational closure."""

from __future__ import annotations

import json

import numpy as np

from dsmc_v2_contracts import DIAGNOSTIC_NAMES, FEATURE_NAMES, cell_invariants, load_run, validate_run
from dsmc_v2_contracts.io import AI, OI, _vec

from .fit_angular import fit_angular_kernel
from .fit_exchange import fit_exchange_kernel
from .weights import (
    effective_sample_size,
    outcome_attempt_indices,
    outcome_weights,
    propensity_diagnostics,
    proposal_balance_diagnostics,
)


N_BLOCKS = 128
N_SCALARS = 10


def _evaluate(sums: np.ndarray, metadata: dict, bl) -> np.ndarray:
    """Deprecated schema-2.1 estimator retained for reproducible A/B tests."""
    from dsmc_v2_contracts import LEGACY_FEATURE_NAMES
    sums = np.asarray(sums, dtype=float)
    count = len(LEGACY_FEATURE_NAMES)
    if sums.size != N_SCALARS + 4 * count:
        raise ValueError("legacy sufficient-statistics row has the wrong size")
    ntry, nhit, sd, st, se, sb2, sp1, sp2, sp3, sp4 = sums[:N_SCALARS]
    alpha, theta, ar = (float(metadata[name]) for name in
                        ("alpha", "theta", "aspect_ratio"))
    area = float(metadata["proposal_area"])
    sigma = float(metadata.get("collision_cross_section", frozen_cross_section(ar)))
    sigma_ctc = area * nhit / ntry
    scalar = [sigma_ctc, sigma, (sigma_ctc - sigma) / sigma]
    if alpha >= 1.0:
        scalar.extend([np.nan] * 4)
    else:
        mean_gamma = bl.parameters(alpha, ar)["mean_loss_fraction"]
        sbl = mean_gamma * se
        f0, fc = area * st / (sigma * sbl), st / sd
        scalar.extend([f0, fc / (3.0 * theta / (3.0 * theta + 2.0)),
                       fc, area * sd / (sigma * sbl)])
    b2 = sb2 / nhit
    scalar.extend([b2, np.nan, sp1 / nhit, sp2 / nhit, sp3 / nhit, sp4 / nhit])
    start = N_SCALARS
    try_scores = sums[start:start + count]
    delta_scores = sums[start + count:start + 2 * count]
    dtr_scores = sums[start + 2 * count:start + 3 * count]
    energy_scores = sums[start + 3 * count:start + 4 * count]
    if alpha >= 1.0:
        beta = beta_ctc = np.full(count, np.nan)
    else:
        lbl = energy_scores / se
        beta = dtr_scores / st - lbl
        beta_ctc = dtr_scores / st - delta_scores / sd
    return np.concatenate((np.asarray(scalar), beta, beta_ctc, try_scores / ntry))


def frozen_cross_section(aspect_ratio: float, diameter: float = 1.0) -> float:
    """Frozen v1 DSMC clock, retained strictly as an audit value."""
    ar = float(aspect_ratio)
    return float(np.pi * diameter * diameter * (0.32 * ar * ar + 0.694 * ar - 0.0213))


def _check_compatible(runs) -> None:
    if not runs:
        raise ValueError("at least one run directory is required")
    reference = runs[0].metadata
    for key in ("alpha", "theta", "aspect_ratio", "velocity_scale", "omega_scale",
                "proposal_area", "mass", "moi_perpendicular", "ensemble_id"):
        values = np.array([float(run.metadata.get(key, 0)) for run in runs])
        if not np.allclose(values, float(reference.get(key, 0)), rtol=2.0e-12, atol=2.0e-12):
            raise ValueError(f"incompatible shard metadata for {key}: {values.tolist()}")
    for run in runs:
        qa = validate_run(run)
        if qa["status"] != "pass":
            raise ValueError(json.dumps(qa, indent=2))


def _run_events(run) -> dict[str, np.ndarray]:
    outcome = np.asarray(run.outcomes)
    values = outcome["values"]
    indices = outcome_attempt_indices(run)
    attempts = np.asarray(run.attempts)
    av = attempts["values"]
    c1, c2 = _vec(av, AI, "c1")[indices], _vec(av, AI, "c2")[indices]
    vcm = 0.5 * (c1 + c2)
    et_in = float(run.metadata["mass"]) * (
        np.sum((c1 - vcm) ** 2, axis=1) + np.sum((c2 - vcm) ** 2, axis=1))
    total_in = values[:, OI["e_initial"]]
    total_out = total_in - values[:, OI["delta_total"]]
    if np.any(total_in <= 0.0) or np.any(total_out <= 0.0):
        raise ValueError("non-positive collision energy pool")
    z_in = et_in / total_in
    z_el = values[:, OI["et_elastic"]] / total_in
    z_out = values[:, OI["et_inelastic"]] / total_out
    gpre = _vec(values, OI, "ghat_pre")
    gpost = _vec(values, OI, "ghat_post")
    cosine = np.clip(np.einsum("ni,ni->n", gpre, gpost), -1.0, 1.0)
    return {
        "z_in": z_in,
        "z_el": z_el,
        "z_out": z_out,
        "cosine": cosine,
        "weight": outcome_weights(run, normalise=False),
        "block": (outcome["block_id"].astype(int)
                  + (int(run.metadata["seed"]) * 31) % N_BLOCKS) % N_BLOCKS,
    }


def _proposal_invariants(runs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity, omega, axis = [], [], []
    for run in runs:
        values = np.asarray(run.attempts["values"])
        velocity.extend((_vec(values, AI, "c1"), _vec(values, AI, "c2")))
        omega.extend((_vec(values, AI, "omega1"), _vec(values, AI, "omega2")))
        axis.extend((_vec(values, AI, "u1"), _vec(values, AI, "u2")))
    features, diagnostics = cell_invariants(
        np.concatenate(velocity), np.concatenate(omega), np.concatenate(axis),
        float(runs[0].metadata["mass"]), float(runs[0].metadata["moi_perpendicular"]))
    return features, diagnostics, np.concatenate(velocity)


def _fit(events: dict[str, np.ndarray], allow_joint: bool = True) -> dict:
    weight = events["weight"]
    weight = weight * len(weight) / np.sum(weight)
    energy = fit_exchange_kernel(events["z_in"], events["z_out"], weight)
    angular = fit_angular_kernel(events["cosine"], events["z_out"], weight,
                                 allow_joint=allow_joint)
    return {"energy": energy, "angular": angular}


def _bootstrap(events: dict[str, np.ndarray], count: int, seed: int) -> dict:
    if count <= 0:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in
        ("p_exch", "reset_mean", "reset_second_moment", "lambda1", "lambda2",
         "eta1", "eta2", "rho_z_cosine")}
    for _ in range(count):
        chosen = rng.integers(0, N_BLOCKS, N_BLOCKS)
        multiplicity = np.bincount(chosen, minlength=N_BLOCKS)
        selected_weight = events["weight"] * multiplicity[events["block"]]
        mask = selected_weight > 0.0
        sample = {key: value[mask] for key, value in events.items() if key != "block"}
        sample["weight"] = selected_weight[mask]
        try:
            fit = _fit(sample, allow_joint=False)
        except (ValueError, np.linalg.LinAlgError):
            continue
        for name in values:
            source = fit["energy"] if name in fit["energy"] else fit["angular"]
            values[name].append(float(source[name]))
    minimum = max(20, count // 2)
    if count and min(map(len, values.values())) < minimum:
        raise ValueError("too few valid bootstrap replicates")
    output = {}
    for name, sample in values.items():
        sample = np.asarray(sample)
        output[name] = {
            "standard_error": float(np.std(sample, ddof=1)),
            "ci_low": float(np.quantile(sample, 0.025)),
            "ci_high": float(np.quantile(sample, 0.975)),
        }
    return output


def estimate_node(run_directories, bl=None, n_bootstrap: int = 200,
                  bootstrap_seed: int = 20260902) -> dict:
    """Estimate one (alpha, theta, AR, ensemble) node.

    ``bl`` remains an accepted argument for command-line compatibility.  It
    is intentionally unused: the CTC fit transfers only the surviving energy
    partition, while the existing BL model remains authoritative for loss.
    """
    runs = [load_run(path) for path in run_directories]
    _check_compatible(runs)
    parts = [_run_events(run) for run in runs]
    events = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    fitted = _fit(events)
    uncertainty = _bootstrap(events, int(n_bootstrap), int(bootstrap_seed))
    features, diagnostics, velocity = _proposal_invariants(runs)
    weight = events["weight"]
    ess = effective_sample_size(weight)
    propensity_rows = [propensity_diagnostics(run) for run in runs]
    propensity_pass = all(row["pass"] for row in propensity_rows)
    balance_rows = [proposal_balance_diagnostics(run) for run in runs]
    balance_pass = all(row["pass"] for row in balance_rows)
    energy = fitted["energy"]
    angular = fitted["angular"]
    alpha = float(runs[0].metadata["alpha"])
    elastic_pass = True
    if np.isclose(alpha, 1.0):
        elastic_pass = (abs(energy["reset_mean"] - 0.5) <= 0.02
                        and abs(energy["reset_second_moment"] - 0.3) <= 0.02)
    qa = {
        "propensity_pass": propensity_pass,
        "proposal_balance_pass": balance_pass,
        "ess_fraction": ess / len(weight),
        "ess_pass": bool(ess >= 0.5 * len(weight)),
        "energy_projection_pass": bool(energy["projection_residual"] < 1.0e-6),
        "angular_projection_pass": bool(angular["projection_residual"] < 1.0e-6),
        "model_form_pass": bool(energy["model_form_pass"]),
        "elastic_pass": bool(elastic_pass),
    }
    qa["sentinel_pass"] = bool(all(qa[name] for name in (
        "propensity_pass", "proposal_balance_pass", "ess_pass", "energy_projection_pass",
        "angular_projection_pass", "model_form_pass", "elastic_pass")))
    metadata = runs[0].metadata
    return {
        "schema_version": "2.2.0",
        "alpha": alpha,
        "theta": float(metadata["theta"]),
        "aspect_ratio": float(metadata["aspect_ratio"]),
        "ensemble_id": int(metadata.get("ensemble_id", 0)),
        "source_schema_versions": sorted({run.metadata["source_schema_version"] for run in runs}),
        "source_runs": [str(run.directory.resolve()) for run in runs],
        "n_attempts": int(sum(len(run.attempts) for run in runs)),
        "n_outcomes": int(len(events["z_out"])),
        "proposal_features": dict(zip(FEATURE_NAMES, features.tolist())),
        "proposal_diagnostics": dict(zip(DIAGNOSTIC_NAMES, diagnostics.tolist())),
        "energy": energy,
        "angular": angular,
        "uncertainty": uncertainty,
        "measure": {
            "weight_definition": "inverse_projected_excluded_area",
            "ess": ess,
            "ess_fraction": ess / len(weight),
            "propensity": propensity_rows,
            "proposal_balance": balance_rows,
            "frozen_cross_section_audit": frozen_cross_section(float(metadata["aspect_ratio"])),
            "mean_center_of_mass_velocity": np.mean(velocity, axis=0).tolist(),
        },
        "qa": qa,
    }
