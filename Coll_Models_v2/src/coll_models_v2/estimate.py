"""Node-wise estimation for the BL-compatible variational closure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import DIAGNOSTIC_NAMES, FEATURE_NAMES, cell_invariants, load_run, validate_run
from dsmc_v2_contracts.io import AI, OI, _vec

from .fit_angular import fit_angular_kernel
from .fit_exchange import fit_exchange_kernel
from .weights import (
    DEFAULT_OFFSETS,
    RELATIVE_BIAS_TOLERANCE,
    effective_sample_size,
    incoming_partition_diagnostics,
    kinematic_propensity,
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


PROPENSITY_CACHE = Path("results/closure_estimates/.propensity_cache")


def _run_propensity(run, offsets: int | None,
                    cache: Path | None = PROPENSITY_CACHE) -> np.ndarray | None:
    """Acceptance probability per proposal, or None to keep the static weight.

    The integral is deterministic given the shard and the offset count, and it
    dominates the cost of a node, so it is cached on disk. The key carries the
    shard identity and its byte size, so a regenerated shard misses the cache.
    """
    if offsets is None:
        return None
    key = None
    if cache is not None:
        directory = Path(run.directory).resolve()
        size = (directory / "attempts_v2.bin").stat().st_size
        key = cache / f"{directory.name}_{size}_{int(offsets)}.npy"
        if key.is_file():
            stored = np.load(key)
            if len(stored) == len(run.attempts):
                return stored
    value = kinematic_propensity(run, offsets=int(offsets))
    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        np.save(key, value)
    return value


def _run_events(run, propensity=None, offsets: int = DEFAULT_OFFSETS) -> dict[str, np.ndarray]:
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
        "loss": values[:, OI["delta_total"]] / total_in,
        "cosine": cosine,
        "weight": outcome_weights(run, normalise=False,
                                  propensity=propensity, offsets=offsets),
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


def _fit(events: dict[str, np.ndarray], allow_joint: bool = True,
         model_form: bool = True, initial: np.ndarray | None = None) -> dict:
    weight = events["weight"]
    weight = weight * len(weight) / np.sum(weight)
    energy = fit_exchange_kernel(events["z_in"], events["z_out"], weight,
                                 loss=events.get("loss"), model_form=model_form,
                                 initial=initial)
    angular = fit_angular_kernel(events["cosine"], events["z_out"], weight,
                                 allow_joint=allow_joint)
    return {"energy": energy, "angular": angular}


def _energy_parameters(energy: dict) -> np.ndarray:
    """Natural parameters in the order the fitted kernel expects."""
    if energy.get("kernel_form") == "sinkhorn_bridge_v2":
        # (memory, tilt...); the tilt is empty in the elastic block.
        values = [energy["lambda3"]]
        if not energy.get("elastic_block"):
            values += [energy["lambda1"], energy["lambda2"]]
            if energy.get("loss_covariate_deployed"):
                values.append(energy["lambda4"])
        return np.asarray(values, dtype=float)
    values = [energy["lambda1"], energy["lambda2"], energy["lambda3"]]
    if energy.get("loss_covariate_deployed"):
        values.append(energy["lambda4"])
    return np.asarray(values, dtype=float)


def _bootstrap(events: dict[str, np.ndarray], count: int, seed: int,
               initial: np.ndarray | None = None) -> dict:
    if count <= 0:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in
        ("p_exch", "reset_mean", "reset_second_moment", "mean_partition_out",
         "lambda1", "lambda2", "lambda3", "lambda4",
         "eta1", "eta2", "rho_z_cosine")}
    for _ in range(count):
        chosen = rng.integers(0, N_BLOCKS, N_BLOCKS)
        multiplicity = np.bincount(chosen, minlength=N_BLOCKS)
        selected_weight = events["weight"] * multiplicity[events["block"]]
        mask = selected_weight > 0.0
        sample = {key: value[mask] for key, value in events.items() if key != "block"}
        sample["weight"] = selected_weight[mask]
        try:
            fit = _fit(sample, allow_joint=False, model_form=False, initial=initial)
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
                  bootstrap_seed: int = 20260902,
                  propensity_offsets: int | None = DEFAULT_OFFSETS) -> dict:
    """Estimate one (alpha, theta, AR, ensemble) node.

    ``bl`` remains an accepted argument for command-line compatibility.  It
    is intentionally unused: the CTC fit transfers only the surviving energy
    partition, while the existing BL model remains authoritative for loss.
    """
    runs = [load_run(path) for path in run_directories]
    _check_compatible(runs)
    propensities = [_run_propensity(run, propensity_offsets) for run in runs]
    offsets = int(propensity_offsets or DEFAULT_OFFSETS)
    parts = [_run_events(run, propensity, offsets)
             for run, propensity in zip(runs, propensities)]
    events = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    fitted = _fit(events)
    uncertainty = _bootstrap(events, int(n_bootstrap), int(bootstrap_seed),
                             initial=_energy_parameters(fitted["energy"]))
    features, diagnostics, velocity = _proposal_invariants(runs)
    weight = events["weight"]
    ess = effective_sample_size(weight)
    propensity_rows = [propensity_diagnostics(run, propensity)
                       for run, propensity in zip(runs, propensities)]
    propensity_pass = all(row["pass"] for row in propensity_rows)
    balance_rows = [proposal_balance_diagnostics(run, propensity, offsets)
                    for run, propensity in zip(runs, propensities)]
    balance_pass = all(row["pass"] for row in balance_rows)
    partition_rows = [incoming_partition_diagnostics(run, propensity, offsets)
                      for run, propensity in zip(runs, propensities)]
    partition_pass = all(row["pass"] for row in partition_rows)
    energy = fitted["energy"]
    angular = fitted["angular"]
    alpha = float(runs[0].metadata["alpha"])
    # Elastic gate: an elastic exchange kernel must drive the partition to
    # equipartition, so its invariant law has to be Beta(2,2) -- mean 1/2,
    # second moment 3/10 -- at every theta and aspect ratio.
    #
    # The tolerance is the looser of three bootstrap standard errors and a flat
    # 2 percent, for the same reason the propensity gate is: a pure
    # significance test necessarily tightens as the event count grows, and a
    # pure absolute test ignores how well the node is actually resolved. A node
    # whose kernel is nearly the identity has a weakly identified invariant law
    # and should be judged on its own error bar.
    elastic_pass, elastic_detail = True, None
    if np.isclose(alpha, 1.0):
        elastic_detail = []
        for name, target in (("reset_mean", 0.5), ("reset_second_moment", 0.3)):
            value = energy["stationary_mean" if name == "reset_mean"
                          else "stationary_second_moment"]
            error = uncertainty.get(name, {}).get("standard_error")
            deviation = abs(value - target)
            allowed = max(3.0 * error, RELATIVE_BIAS_TOLERANCE) if error is not None \
                else RELATIVE_BIAS_TOLERANCE
            elastic_detail.append({
                "quantity": name, "value": float(value), "target": target,
                "deviation": float(deviation), "standard_error": error,
                "sigma": float(deviation / error) if error else None,
                "allowed": float(allowed), "pass": bool(deviation <= allowed),
            })
        elastic_pass = all(row["pass"] for row in elastic_detail)
    qa = {
        "propensity_pass": propensity_pass,
        "proposal_balance_pass": balance_pass,
        "ess_fraction": ess / len(weight),
        "ess_pass": bool(ess >= 0.5 * len(weight)),
        "energy_projection_pass": bool(energy["projection_residual"] < 1.0e-6),
        "angular_projection_pass": bool(angular["projection_residual"] < 1.0e-6),
        "model_form_pass": bool(energy["model_form_pass"]),
        "memory_diagnostic_pass": bool(energy["memory_diagnostic_pass"]),
        "incoming_partition_pass": bool(partition_pass),
        "elastic_pass": bool(elastic_pass),
    }
    qa["sentinel_pass"] = bool(all(qa[name] for name in (
        "propensity_pass", "proposal_balance_pass", "ess_pass", "energy_projection_pass",
        "angular_projection_pass", "model_form_pass", "memory_diagnostic_pass",
        "incoming_partition_pass", "elastic_pass")))
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
            "weight_definition": ("inverse_kinematic_propensity"
                                  if propensity_offsets is not None
                                  else "inverse_projected_excluded_area"),
            "propensity_offsets": propensity_offsets,
            "ess": ess,
            "ess_fraction": ess / len(weight),
            "propensity": propensity_rows,
            "proposal_balance": balance_rows,
            "incoming_partition": partition_rows,
            "elastic_limit": elastic_detail,
            "frozen_cross_section_audit": frozen_cross_section(float(metadata["aspect_ratio"])),
            "mean_center_of_mass_velocity": np.mean(velocity, axis=0).tolist(),
        },
        "qa": qa,
    }
