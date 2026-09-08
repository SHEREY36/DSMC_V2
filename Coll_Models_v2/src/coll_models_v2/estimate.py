"""Node-wise estimation for the BL-compatible variational closure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import DIAGNOSTIC_NAMES, FEATURE_NAMES, cell_invariants, load_run, validate_run
from dsmc_v2_contracts.io import AI, OI, _vec

from .fit_angular import fit_angular_kernel
from .fit_exchange import fit_exchange_kernel
from .projections import bridge_stationary
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
NODE_ESTIMATE_CONTRACT = "node_estimate_v3_energy_weighted_incoming_law"


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


MEASURE = "collision"
# "collision": fit on the raw accepted hits. These ARE the physical collision
#   ensemble -- pairs enter in proportion to how often they actually collide --
#   and they satisfy the elastic invariant to 4e-4.
# "proposal": divide out the acceptance to recover the orientation-isotropic
#   proposal ensemble. Retained for A/B only: it breaks the elastic invariant by
#   -0.0117 at AR 3 because the weight correlates with the outgoing partition
#   through the same geometry that sets the acceptance.


def _run_events(run, propensity=None, offsets: int = DEFAULT_OFFSETS,
                measure: str = MEASURE,
                attempt_weight: np.ndarray | None = None) -> dict[str, np.ndarray]:
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
    weight = (np.ones(len(values)) if measure == "collision"
              else outcome_weights(run, normalise=False,
                                   propensity=propensity, offsets=offsets))
    if attempt_weight is not None:
        attempt_weight = np.asarray(attempt_weight, dtype=float)
        if attempt_weight.shape != (len(attempts),) \
                or np.any(~np.isfinite(attempt_weight)) \
                or np.any(attempt_weight < 0.0) \
                or not np.any(attempt_weight > 0.0):
            raise ValueError("attempt importance weights are invalid")
        # The accepted collision law contains the same geometric hit factor in
        # baseline and target ensembles, so only the incoming pair density
        # ratio remains.  Map it by the recorded attempt index; never assume
        # outcomes happen to retain attempt-row order.
        weight = weight * attempt_weight[indices]
    return {
        "z_in": z_in,
        "z_el": z_el,
        "z_out": z_out,
        "loss": values[:, OI["delta_total"]] / total_in,
        "energy": total_in,
        "cosine": cosine,
        "weight": weight,
        "block": (outcome["block_id"].astype(int)
                  + (int(run.metadata["seed"]) * 31) % N_BLOCKS) % N_BLOCKS,
    }


def _systematic(weight: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Low-variance resampling: one stratified pass, not a multinomial draw."""
    weight = weight / np.sum(weight)
    positions = (np.random.default_rng(seed).random() + np.arange(count)) / count
    return np.searchsorted(np.cumsum(weight), positions, side="right").clip(
        0, len(weight) - 1)


def _proposal_invariants(runs, cell_measure: bool = False
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invariants of the proposal ensemble.

    ``cell_measure`` de-biases the collision-flux weighting. The generator draws
    the relative speed from p(g) ~ g^3 exp(-g^2/4kT) while a gas cell holds
    p(g) ~ g^2 exp(-g^2/4kT): exactly one power of g. Left in, the attempt
    ensemble reports a2_tr = -0.0322 for a shard whose underlying gas is
    Maxwellian; weighting by 1/|g| returns +0.00007.

    That matters because the excitation coefficients answer "how does the kernel
    move when the CELL carries an invariant". Regressing against the
    collision-attempt marginal instead would push a fixed offset straight into
    beta.
    """
    velocity, omega, axis, speed = [], [], [], []
    for run in runs:
        values = np.asarray(run.attempts["values"])
        c1, c2 = _vec(values, AI, "c1"), _vec(values, AI, "c2")
        velocity.extend((c1, c2))
        omega.extend((_vec(values, AI, "omega1"), _vec(values, AI, "omega2")))
        axis.extend((_vec(values, AI, "u1"), _vec(values, AI, "u2")))
        relative = np.linalg.norm(c1 - c2, axis=1)
        speed.extend((relative, relative))
    velocity = np.concatenate(velocity)
    omega, axis = np.concatenate(omega), np.concatenate(axis)
    mass = float(runs[0].metadata["mass"])
    inertia = float(runs[0].metadata["moi_perpendicular"])
    if cell_measure:
        pick = _systematic(1.0 / np.maximum(np.concatenate(speed), 1.0e-30),
                           len(velocity), seed=0x0CE11)
        velocity, omega, axis = velocity[pick], omega[pick], axis[pick]
    features, diagnostics = cell_invariants(velocity, omega, axis, mass, inertia)
    return features, diagnostics, velocity


def equilibrium_anchor(run_directories, measure: str = MEASURE,
                       propensity_offsets: int | None = DEFAULT_OFFSETS) -> tuple:
    """Reference law, measured on an elastic equipartitioned shard.

    The bridge is reversible with respect to this law, so it has to be the
    EQUILIBRIUM -- a property of the aspect ratio under collision-ensemble
    selection -- not the law of whatever theta a node was generated at. Pass
    the result to every node sharing that aspect ratio.
    """
    from .fit_exchange import measure_anchor
    runs = [load_run(directory) for directory in run_directories]
    offsets = int(propensity_offsets or DEFAULT_OFFSETS)
    parts = [_run_events(run, _run_propensity(run, propensity_offsets), offsets, measure)
             for run in runs]
    events = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    weight = events["weight"] * len(events["weight"]) / np.sum(events["weight"])
    return measure_anchor(events["z_in"], weight)


def energy_anchor_moments(energy: dict) -> tuple[float, float]:
    """First two moments of the law the fitted kernel is anchored on."""
    anchor = (float(energy.get("anchor_c1", 0.0)), float(energy.get("anchor_c2", 0.0)))
    nodes, mass = bridge_stationary(np.array([0.0]), 0.0, 128, anchor)
    return float(mass @ nodes), float(mass @ (nodes * nodes))


def _fit(events: dict[str, np.ndarray], allow_joint: bool = True,
         model_form: bool = True, initial: np.ndarray | None = None,
         anchor: tuple | None = None) -> dict:
    weight = events["weight"]
    weight = weight * len(weight) / np.sum(weight)
    energy = fit_exchange_kernel(events["z_in"], events["z_out"], weight,
                                 loss=events.get("loss"), model_form=model_form,
                                 initial=initial, anchor=anchor)
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
               initial: np.ndarray | None = None, anchor: tuple | None = None) -> dict:
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
            fit = _fit(sample, allow_joint=False, model_form=False, initial=initial,
                   anchor=anchor)
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
                  propensity_offsets: int | None = DEFAULT_OFFSETS,
                  measure: str = MEASURE,
                  anchor: tuple | None = None,
                  attempt_weights: list[np.ndarray] | None = None,
                  cell_features_override: np.ndarray | None = None,
                  ensemble_id_override: int | None = None,
                  excitation: dict | None = None) -> dict:
    """Estimate one (alpha, theta, AR, ensemble) node.

    ``bl`` remains an accepted argument for command-line compatibility.  It
    is intentionally unused: the CTC fit transfers only the surviving energy
    partition, while the existing BL model remains authoritative for loss.

    ``attempt_weights`` optionally supplies a target-to-baseline incoming-pair
    density ratio for each shard.  It is joined to accepted outcomes by their
    recorded attempt keys, enabling exact importance-sampled excitation without
    pretending that a reweighted sample was freshly generated CTC data.
    """
    runs = [load_run(path) for path in run_directories]
    _check_compatible(runs)
    if attempt_weights is None:
        attempt_weights = [None] * len(runs)
    elif len(attempt_weights) != len(runs):
        raise ValueError("one attempt-weight array is required per input shard")
    propensities = [_run_propensity(run, propensity_offsets) for run in runs]
    offsets = int(propensity_offsets or DEFAULT_OFFSETS)
    parts = [_run_events(run, propensity, offsets, measure, importance)
             for run, propensity, importance in
             zip(runs, propensities, attempt_weights)]
    events = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    fitted = _fit(events, anchor=anchor)
    # The node's OWN incoming law. This is no longer the bridge's reference
    # measure -- that is the shared equilibrium -- but it is the law the
    # stability gate must average the mean map over, because it is what the
    # kernel is actually fed at this theta.
    from .fit_exchange import measure_anchor
    _w = events["weight"] * len(events["weight"]) / np.sum(events["weight"])
    incoming_c1, incoming_c2 = measure_anchor(events["z_in"], _w)
    # Modal energy drift is driven by <E z>, not <z>: a collision carrying twice
    # the energy moves the gas twice as far. The gate averages the mean map over
    # THIS law, so it has to be measured with the energy weight rather than
    # relabelled after the fact.
    _we = _w * events["energy"]
    energy_c1, energy_c2 = measure_anchor(events["z_in"], _we)
    # The anchor must reach the replicates too, or each one re-measures the
    # reference law from its own resample and the bootstrap reports the spread
    # of a different estimator than the point fit.
    uncertainty = _bootstrap(events, int(n_bootstrap), int(bootstrap_seed),
                             initial=_energy_parameters(fitted["energy"]),
                             anchor=anchor)
    features, diagnostics, velocity = _proposal_invariants(runs)
    cell_features_value, _, _ = _proposal_invariants(runs, cell_measure=True)
    if cell_features_override is not None:
        override = np.asarray(cell_features_override, dtype=float)
        if override.shape != (len(FEATURE_NAMES),) or np.any(~np.isfinite(override)):
            raise ValueError("cell feature override has the wrong shape or is non-finite")
        cell_features_value = override
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
        # The elastic kernel must leave the *measured* incoming law alone. On
        # the proposal ensemble that law is Beta(2,2); on the physical collision
        # ensemble it is not, so the target is the anchor the fit measured.
        anchor_first, anchor_second = energy_anchor_moments(energy)
        for name, target in (("reset_mean", anchor_first),
                             ("reset_second_moment", anchor_second)):
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
        # This is deliberately separate from the CTC record schema.  A node
        # JSON can still be schema 2.2 while having been produced before the
        # energy-weighted incoming law and cell-measure features existed.
        # Artifact construction therefore keys on this semantic contract,
        # rather than accepting an old estimate because its record schema is
        # readable.
        "estimator_contract": NODE_ESTIMATE_CONTRACT,
        "alpha": alpha,
        "theta": float(metadata["theta"]),
        "aspect_ratio": float(metadata["aspect_ratio"]),
        "ensemble_id": int(metadata.get("ensemble_id", 0) if ensemble_id_override is None
                           else ensemble_id_override),
        "source_schema_versions": sorted({run.metadata["source_schema_version"] for run in runs}),
        "source_runs": [str(run.directory.resolve()) for run in runs],
        "n_attempts": int(sum(len(run.attempts) for run in runs)),
        "n_outcomes": int(len(events["z_out"])),
        "proposal_features": dict(zip(FEATURE_NAMES, features.tolist())),
        # The measure the DSMC cell actually carries. beta must be regressed
        # against this, never against the collision-attempt marginal above.
        "cell_features": dict(zip(FEATURE_NAMES, cell_features_value.tolist())),
        "incoming_law": {"c1": incoming_c1, "c2": incoming_c2},
        "incoming_law_energy": {"c1": energy_c1, "c2": energy_c2},
        "proposal_diagnostics": dict(zip(DIAGNOSTIC_NAMES, diagnostics.tolist())),
        "energy": energy,
        "angular": angular,
        "uncertainty": uncertainty,
        "measure": {
            "weight_definition": ("inverse_kinematic_propensity"
                                  if propensity_offsets is not None
                                  else "inverse_projected_excluded_area"),
            "propensity_offsets": propensity_offsets,
            "measure_ensemble": measure,
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
        "excitation": excitation,
    }
