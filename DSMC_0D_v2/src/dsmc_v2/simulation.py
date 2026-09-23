"""Conservative public HCS DSMC driver based on the proven v1 time march."""

from __future__ import annotations

import math
import resource
import time as wallclock
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import (
    FEATURE_NAMES,
    cell_features,
    cell_features_with_domain,
    legacy_cell_features,
)

from .artifact import MicroscopicClosure, VariationalClosure
from .kernel import SpherocylinderKernel
from .legacy_models import FrozenLossModel, LegacyModels
from .non_gaussian import NonGaussianDiagnostics
from .ntc import NTCWorkspace, candidate_count
from .particle import particle_parameters
from .pressure import accumulate_pij_c, compute_pij_k, normalise_pij_c
from .state import initialize_particles


def runtime_gate_status(diagnostics: dict) -> dict:
    """Evaluate the variational production limits without hiding failures."""
    reasons = []
    if int(diagnostics["negative_energy_repairs"]) != 0:
        reasons.append("negative_energy_repairs_not_zero")
    if "correction_fallback_fraction_in_evaluation_window" in diagnostics:
        fallback_fraction = float(
            diagnostics["correction_fallback_fraction_in_evaluation_window"])
        if fallback_fraction >= 1.0e-2:
            reasons.append(
                "correction_fallback_fraction_in_evaluation_window_not_below_0.01")
    elif float(diagnostics.get("out_of_domain_fraction", 0.0)) >= 1.0e-3:
        # Compatibility for older diagnostics that predate safe fallback.
        reasons.append("out_of_domain_fraction_not_below_0.001")
    if float(diagnostics["closure_overhead_fraction"]) >= 0.15:
        reasons.append("closure_overhead_fraction_not_below_0.15")
    if int(diagnostics.get("energy_axis_clamps", 0)) != 0:
        reasons.append("energy_axis_clamps_not_zero")
    if int(diagnostics.get("energy_monotonic_repairs", 0)) != 0:
        reasons.append("energy_monotonic_repairs_not_zero")
    if float(diagnostics.get(
            "maximum_bulk_to_thermal_temperature_ratio", 0.0)) >= 1.0e-12:
        reasons.append("bulk_to_thermal_temperature_ratio_not_below_1e-12")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "limits": {
            "negative_energy_repairs": 0,
            "correction_fallback_fraction_in_evaluation_window_exclusive_maximum":
                1.0e-2,
            "closure_overhead_fraction_exclusive_maximum": 0.15,
            "energy_axis_clamps": 0,
            "energy_monotonic_repairs": 0,
            "bulk_to_thermal_temperature_ratio_exclusive_maximum": 1.0e-12,
        },
    }


def _closure_from_config(config: dict):
    closure_config = config.get("microscopic_closure", {})
    routing = closure_config.get("routing", "legacy_rank0")
    angular = closure_config.get("angular", "legacy")
    if routing not in ("legacy_rank0", "ctc_moment16", "variational_v2"):
        raise ValueError("routing must be legacy_rank0, ctc_moment16, or variational_v2")
    if angular not in ("legacy", "ctc_vss_rank2", "variational_v2"):
        raise ValueError("angular must be legacy, ctc_vss_rank2, or variational_v2")
    if routing == "legacy_rank0" and angular == "legacy":
        return routing, angular, None
    if routing == "variational_v2" or angular == "variational_v2":
        if routing != "variational_v2" or angular != "variational_v2":
            raise ValueError("variational_v2 energy and angle must be enabled together")
        return routing, angular, VariationalClosure(
            closure_config["artifact"],
            bool(closure_config.get("invariant_corrections", True)))
    return routing, angular, MicroscopicClosure(
        closure_config["routing_artifact"], closure_config["vss_artifact"],
        closure_config["rotational_direction_artifact"])


def _sphere_collision(state, p1, p2, normal, v1, v2, cr, alpha) -> int:
    coefficient = 0.5 * (1.0 + alpha)
    state.velocity[p1] = v1 - coefficient * cr * normal
    state.velocity[p2] = v2 + coefficient * cr * normal
    return 2


def _write_row(handle, time: float, tau: float, state, mass: float) -> None:
    ttr, trot, total = state.temperatures(mass)
    handle.write(f"{time:13.6f} {tau:13.6f} {ttr:13.6f} {trot:13.6f} {total:13.6f}\n")


def _pair_modal_energies(state, p1: int, p2: int, v1: np.ndarray,
                         v2: np.ndarray, mass: float) -> tuple[float, float]:
    """Translational COM-frame and rotational energy of one selected pair."""
    vcom = 0.5 * (v1 + v2)
    etr = 0.5 * mass * (np.dot(v1 - vcom, v1 - vcom)
                        + np.dot(v2 - vcom, v2 - vcom))
    erot = float(state.rotational_energy[p1] + state.rotational_energy[p2])
    return float(etr), erot


def run_simulation(config: dict, seed: int, output_path: str | Path,
                   pressure_path: str | Path | None = None,
                   orientation_path: str | Path | None = None) -> dict:
    """Run one realization while preserving the v1 clock and scalar kernel."""
    flow = config.get("flow", {})
    flow_mode = flow.get("mode", "hcs")
    if flow_mode not in ("hcs", "usf"):
        raise ValueError("flow.mode must be hcs or usf")
    shear_rate = float(flow.get("shear_rate", 0.0))
    np.random.seed(int(seed))
    axis_rng = np.random.default_rng(int(seed) + 0x6A09E667)
    direction_rng = np.random.default_rng(int(seed) + 0xBB67AE85)
    vss_rng = np.random.default_rng(int(seed) + 0x3C6EF372)
    params = particle_parameters(config)
    alpha = float(config["system"]["alpha"])
    ktt, ktr = float(config["system"]["kTt"]), float(config["system"]["kTr"])
    volume = float(np.prod(config["system"]["domain"]))
    count = math.ceil(float(config["system"]["phi"]) * volume / params.volume)
    sphere = bool(config.get("simulation", {}).get("sphere_collision", False))
    state = initialize_particles(count, ktt, ktr, params.mass, params.inertia,
                                 axis_rng, sphere,
                                 isotropic_rotation=(
                                     config.get("microscopic_closure", {}).get("routing")
                                     == "variational_v2"))
    exact_initial_temperatures = bool(config.get("simulation", {}).get(
        "exact_initial_temperatures", False))
    if exact_initial_temperatures and not sphere:
        state.set_modal_temperatures(ktt, ktr, params.mass)
    elastic_limit = config.get("microscopic_closure", {}).get(
        "elastic_limit", "exact_bl")
    if elastic_limit not in ("exact_bl", "closure"):
        raise ValueError("microscopic_closure.elastic_limit must be exact_bl or closure")
    if not sphere and alpha >= 1.0 and elastic_limit == "exact_bl":
        # alpha=1 is a singular point of the fitted closure (no loss, and the
        # near-sphere exchange is too slow to reach equipartition in any
        # affordable run). It gets its own exact block and never queries the
        # artifact, so it cannot fail closed at the hull boundary either.
        routing, angular, closure = "elastic_bl", "legacy", None
    else:
        routing, angular, closure = _closure_from_config(config)
    if sphere:
        models = kernel = None
    else:
        dissipation = config["preprocessing"]["dissipation"]
        model_root = config["preprocessing"].get("model_root", "models")
        if routing in ("variational_v2", "elastic_bl"):
            models = FrozenLossModel(model_root, float(dissipation["beta_a"]),
                                     float(dissipation["beta_b"]))
        else:
            models = LegacyModels(model_root, params.aspect_ratio,
                                  float(dissipation["beta_a"]),
                                  float(dissipation["beta_b"]))
        kernel = SpherocylinderKernel(
            params, models, alpha, float(dissipation["beta_a"]),
            float(dissipation["beta_b"]), float(config["system"].get("C_alpha", 1.0)),
            closure, routing, angular, direction_rng, vss_rng,
            float(config["time"].get("equilibration_time", 0.0)),
            bool(config.get("simulation", {}).get("use_isotropic_eps", True)))

    dt = float(config["time"]["dt"])
    dtau = float(config["time"]["dtau"])
    end_time = float(config["time"]["t_end"])
    tau_end = config["time"].get("tau_end")
    tau_end = None if tau_end is None else float(tau_end)
    hcs_rescale = bool(config.get("simulation", {}).get(
        "hcs_rescale_temperature", False))
    if hcs_rescale and flow_mode != "hcs":
        raise ValueError("HCS temperature rescaling is only valid for flow.mode=hcs")
    initial_ttr, _, initial_total = state.temperatures(params.mass)
    rescale_reference = initial_ttr if sphere else initial_total
    # A two-temperature spherocylinder can transfer rotational energy into
    # translation even while its total HCS energy cools.  The old majorant was
    # initialized from kTt alone and was therefore knowingly too small for a
    # low-theta stability start.  Bound the largest translational temperature
    # available from the fixed total modal energy, 3*Ttr + 2*Trot.
    # Preserve the seeded v1 collision sequence for the explicitly requested
    # legacy kernel.  Production angular-evidence (and other nonlegacy)
    # kernels use the conservative two-temperature bound needed by low-theta
    # starts; changing the legacy majorant would consume a different number of
    # candidates and break its reproducibility contract.
    vrmax_temperature_bound = (
        ktt if sphere or routing == "legacy_rank0"
        else ktt + (2.0 / 3.0) * ktr
    )
    vrmax = 5.0 * np.sqrt(2.0) * np.sqrt(vrmax_temperature_bound / params.mass)
    initial_vrmax = vrmax
    # A rescaled HCS is stationary, so the majorant in thermal units must stay
    # O(5-10).  Anything this far out is a runaway, not a velocity tail: fail
    # fast with a diagnosis instead of growing the candidate arrays to OOM.
    majorant_runaway_ratio = float(config.get("simulation", {}).get(
        "majorant_runaway_ratio", 25.0))
    # A candidate list much larger than the particle population is both a
    # memory hazard and a time-discretisation warning: many accepted pairs can
    # then reuse particles whose velocities have already changed in this
    # step.  Bound the allocation *before* NTCWorkspace grows.  The default is
    # intentionally generous for existing runs and can be tightened by
    # campaign fixtures after a pilot has measured the normal workload.
    max_ntc_candidates = int(config.get("simulation", {}).get(
        "max_ntc_candidates_per_step", max(100_000, 50 * count)))
    if max_ntc_candidates <= 0:
        raise ValueError("simulation.max_ntc_candidates_per_step must be positive")
    ntc_candidates = ntc_violations = ntc_steps = 0
    ntc_collision_pairs = ntc_repeated_particle_pairs = 0
    peak_ntc_candidates = 0
    time, collisions, output_index = 0.0, 0, 0
    workspace = NTCWorkspace(capacity=1024, seed=seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_ftr, minimum_ftr = -np.inf, np.inf
    closure_config = config.get("microscopic_closure", {})
    closure_state_update_cpp = float(closure_config.get("state_update_cpp", 0.0))
    if not np.isfinite(closure_state_update_cpp) or closure_state_update_cpp < 0.0:
        raise ValueError("microscopic_closure.state_update_cpp must be finite and nonnegative")
    closure_state_interval = max(1, math.ceil(closure_state_update_cpp * count))
    closure_state_next_collision = 0
    closure_state_updates = 0
    closure_state_seconds = 0.0
    correction_fallback_queries = 0
    evaluation_closure_queries = 0
    evaluation_correction_fallback_queries = 0
    non_gaussian_config = config.get("diagnostics", {}).get("non_gaussian", {})
    evaluation_start_tau = float(non_gaussian_config.get(
        "sample_start_tau", 0.0))
    march_started = wallclock.perf_counter()
    pressure_path = Path(pressure_path) if pressure_path is not None else None
    if flow_mode == "usf" and pressure_path is None:
        pressure_path = output_path.with_name(output_path.stem + "_pressure.txt")
    if pressure_path is not None:
        pressure_path.parent.mkdir(parents=True, exist_ok=True)
    orientation_path = (Path(orientation_path)
                        if orientation_path is not None else None)
    if flow_mode == "usf" and orientation_path is None:
        orientation_path = output_path.with_name(
            output_path.stem + "_orientation.txt")
    if orientation_path is not None:
        orientation_path.parent.mkdir(parents=True, exist_ok=True)
    non_gaussian = NonGaussianDiagnostics(
        config, output_path, count, params.mass, params.inertia, sphere)
    theta_minimum, theta_maximum = np.inf, -np.inf
    minimum_theta_hull_log_margin = np.inf
    maximum_bulk_to_thermal_temperature_ratio = 0.0
    theta_guard_fraction = float(config.get("simulation", {}).get(
        "closure_theta_guard_fraction", 0.0))
    if not 0.0 <= theta_guard_fraction < 0.5:
        raise ValueError("simulation.closure_theta_guard_fraction must lie in [0, 0.5)")
    theta_hull_bounds = (closure.physical_theta_bounds(alpha, params.aspect_ratio)
                         if isinstance(closure, VariationalClosure) else None)
    pressure_accumulator = np.zeros((3, 3)) if flow_mode == "usf" else None
    last_pressure_time = 0.0
    audit_enabled = bool(config.get("diagnostics", {}).get("collision_audit", False))
    audit = {
        "post_ntc_pairs": 0, "accepted_pairs": 0,
        "post_ntc_z_sum": 0.0, "post_ntc_energy_sum": 0.0,
        "post_ntc_ez_sum": 0.0, "accepted_z_sum": 0.0,
        "accepted_energy_sum": 0.0, "accepted_ez_sum": 0.0,
        "outgoing_energy_sum": 0.0, "outgoing_ez_sum": 0.0,
        "loss_sum": 0.0, "energy_loss_sum": 0.0,
        "routing_loss_sum": 0.0, "drift_energy_sum": 0.0,
        "expected_drift_energy_sum": 0.0, "theta_energy_sum": 0.0,
    }
    feature_count = 0
    feature_sum = np.zeros(len(FEATURE_NAMES))
    feature_min = np.full(feature_sum.shape, np.inf)
    feature_max = np.full(feature_sum.shape, -np.inf)
    domain_feature_sum = np.zeros(len(FEATURE_NAMES))
    domain_feature_min = np.full(feature_sum.shape, np.inf)
    domain_feature_max = np.full(feature_sum.shape, -np.inf)
    pressure_context = pressure_path.open("w", buffering=65536) if pressure_path else nullcontext(None)
    orientation_context = (orientation_path.open("w", buffering=65536)
                           if orientation_path else nullcontext(None))
    with (output_path.open("w", buffering=65536) as handle,
          pressure_context as pressure_handle,
          orientation_context as orientation_handle,
          non_gaussian as non_gaussian):
        while time < end_time and (tau_end is None or collisions / count < tau_end
                                   or collisions / count >= output_index * dtau):
            tau = collisions / float(count)
            if tau >= output_index * dtau:
                _write_row(handle, time, tau, state, params.mass)
                handle.flush()   # a killed task keeps its trajectory
                non_gaussian.maybe_sample(time, tau, state)
                if pressure_handle is not None:
                    kinetic = compute_pij_k(state.velocity, params.mass, volume)
                    collisional = normalise_pij_c(
                        pressure_accumulator, time - last_pressure_time, volume)
                    values = [kinetic[0, 0], kinetic[0, 1], kinetic[0, 2],
                              kinetic[1, 1], kinetic[1, 2], kinetic[2, 2],
                              collisional[0, 0], collisional[0, 1], collisional[0, 2],
                              collisional[1, 1], collisional[1, 2], collisional[2, 2]]
                    pressure_handle.write(f"{time:13.6f} {tau:13.6f} "
                                          + " ".join(f"{value:13.6f}" for value in values) + "\n")
                    pressure_accumulator[:] = 0.0
                    last_pressure_time = time
                if orientation_handle is not None:
                    q = state.orientation_tensor()
                    values = [q[0, 0], q[0, 1], q[0, 2],
                              q[1, 1], q[1, 2], q[2, 2]]
                    orientation_handle.write(
                        f"{time:13.6f} {tau:13.6f} "
                        + " ".join(f"{value:13.6f}" for value in values)
                        + "\n")
                output_index += 1
            if flow_mode == "usf":
                state.velocity[:, 0] -= shear_rate * state.velocity[:, 1] * dt
            ttr, trot, _ = state.temperatures(params.mass)
            theta = ttr / trot if trot > 0.0 else 1.0
            theta_minimum = min(theta_minimum, theta)
            theta_maximum = max(theta_maximum, theta)
            if isinstance(closure, VariationalClosure):
                theta_lower, theta_upper = theta_hull_bounds
                log_span = np.log(theta_upper / theta_lower)
                lower_margin = np.log(theta / theta_lower) / log_span
                upper_margin = np.log(theta_upper / theta) / log_span
                theta_margin = min(lower_margin, upper_margin)
                minimum_theta_hull_log_margin = min(
                    minimum_theta_hull_log_margin, theta_margin)
                # A zero guard preserves the artifact's closed hull for all
                # existing callers. HCS-NG explicitly opts into a positive
                # interior margin through its campaign fixture.
                if theta_guard_fraction > 0.0 and theta_margin <= theta_guard_fraction:
                    raise RuntimeError(
                        "HCS closure theta safety margin crossed: "
                        f"theta={theta:.8g}, calibrated=[{theta_lower:.8g},"
                        f" {theta_upper:.8g}], normalized_log_margin="
                        f"{theta_margin:.6g}, guard={theta_guard_fraction:.6g}, "
                        f"tau={collisions / float(count):.3f}. This trajectory "
                        "is aborted_not_stationary; do not extend the hull.")
            if kernel is not None and routing == "ctc_moment16":
                features = legacy_cell_features(state.velocity, state.omega, state.axis,
                                                params.mass, params.inertia, sphere=False)
                ftr = closure.routing_fraction(alpha, theta, params.aspect_ratio, features)
                kernel.set_cell_routing(ftr)
                minimum_ftr, maximum_ftr = min(minimum_ftr, ftr), max(maximum_ftr, ftr)
            elif (kernel is not None and routing == "variational_v2"
                  and (closure_state_update_cpp == 0.0
                       or closure_state_updates == 0
                       or collisions >= closure_state_next_collision)):
                closure_started = wallclock.perf_counter()
                features, domain_features = cell_features_with_domain(
                    state.velocity, state.omega, state.axis,
                    params.mass, params.inertia, sphere=False)
                if audit_enabled:
                    feature_count += 1
                    feature_sum += features
                    feature_min = np.minimum(feature_min, features)
                    feature_max = np.maximum(feature_max, features)
                    domain_feature_sum += domain_features
                    domain_feature_min = np.minimum(
                        domain_feature_min, domain_features)
                    domain_feature_max = np.maximum(
                        domain_feature_max, domain_features)
                closure_alpha = 1.0 if time < kernel.equilibration_time else alpha
                closure_state = closure.kernel_state(
                    closure_alpha, theta, params.aspect_ratio,
                    features, domain_features)
                correction_fallback = bool(
                    closure_state.get("correction_fallback", False))
                correction_fallback_queries += int(correction_fallback)
                if collisions / float(count) >= evaluation_start_tau:
                    evaluation_closure_queries += 1
                    evaluation_correction_fallback_queries += int(
                        correction_fallback)
                kernel.set_cell_variational(closure_state)
                closure_state_seconds += wallclock.perf_counter() - closure_started
                closure_state_updates += 1
                closure_state_next_collision = collisions + closure_state_interval

            if routing == "variational_v2" and not getattr(kernel, "_enhanced", False):
                kernel.set_enhancement(
                    closure.xi_grid, float(np.max(closure.xi_enhancement)))
                kernel._enhanced = True
            inflation = (kernel.candidate_inflation
                         if (kernel is not None and routing == "variational_v2")
                         else 1.0)
            n_candidates = candidate_count(count, params.sigma_c * inflation,
                                           vrmax, volume, dt)
            peak_ntc_candidates = max(peak_ntc_candidates, n_candidates)
            if n_candidates > max_ntc_candidates:
                raise RuntimeError(
                    "NTC candidate ceiling exceeded before allocation: "
                    f"requested={n_candidates}, ceiling={max_ntc_candidates}, "
                    f"vrmax={vrmax:.6g}, inflation={inflation:.6g}, "
                    f"tau={collisions / float(count):.3f}")
            vrmax_temp = 0.0
            ntc_steps += 1
            if n_candidates > 0:
                vrmax_temp, accepted = workspace.screen_candidates(
                    state.velocity, count, n_candidates, vrmax)
                ntc_candidates += n_candidates
                ntc_violations += workspace.last_violations
                # DSMC is a small-dt method: a particle selected for two
                # *actual* collisions in one step sees an already-updated
                # velocity on the second event. Its frequency is therefore a
                # direct and inexpensive time-step quality diagnostic.
                step_collision_particles: set[int] = set()
                for position in accepted:
                    p1, p2 = int(workspace.p1[position]), int(workspace.p2[position])
                    normal = workspace.eij[position].copy()
                    v1, v2 = state.velocity[p1].copy(), state.velocity[p2].copy()
                    vrel = v1 - v2
                    cr = float(np.dot(normal, vrel))
                    if cr < 0.0:
                        normal, cr = -normal, -cr
                    speed = float(np.linalg.norm(vrel))
                    pair_before = None
                    if audit_enabled and not sphere and routing != "elastic_bl":
                        pair_before = _pair_modal_energies(
                            state, p1, p2, v1, v2, params.mass)
                        pair_total = pair_before[0] + pair_before[1]
                        if pair_total > 0.0:
                            pair_z = pair_before[0] / pair_total
                            audit["post_ntc_pairs"] += 1
                            audit["post_ntc_z_sum"] += pair_z
                            audit["post_ntc_energy_sum"] += pair_total
                            audit["post_ntc_ez_sum"] += pair_total * pair_z
                    added = 0
                    if sphere:
                        added = _sphere_collision(
                            state, p1, p2, normal, v1, v2, cr, alpha)
                        collisions += added
                    elif routing == "variational_v2" and not kernel.accept_orientation(
                            state.axis[p1], state.axis[p2],
                            vrel / max(speed, 1.0e-30),
                            state.omega[p1], state.omega[p2], speed, vss_rng):
                        pass          # rejected by the orientation-dependent area
                    else:
                        added = kernel.collide(
                            state, p1, p2, normal, v1, v2, vrel, speed, time, theta)
                        collisions += added
                        if audit_enabled and pair_before is not None and added:
                            etr_i, erot_i = pair_before
                            total_i = etr_i + erot_i
                            etr_f, erot_f = _pair_modal_energies(
                                state, p1, p2, state.velocity[p1],
                                state.velocity[p2], params.mass)
                            total_f = etr_f + erot_f
                            z_in = etr_i / total_i
                            z_out = etr_f / total_f
                            loss = 1.0 - total_f / total_i
                            fitted_loss = float(kernel.cell_variational["fitted_mean_loss"])
                            routing_loss = loss
                            if kernel.mean_loss_fraction > 0.0 and fitted_loss > 0.0:
                                routing_loss *= fitted_loss / kernel.mean_loss_fraction
                            expected_z_out = closure.mean_energy(
                                kernel.cell_variational, z_in, loss,
                                loss_mean=kernel.mean_loss_fraction)
                            audit["accepted_pairs"] += 1
                            audit["accepted_z_sum"] += z_in
                            audit["accepted_energy_sum"] += total_i
                            audit["accepted_ez_sum"] += total_i * z_in
                            audit["outgoing_energy_sum"] += total_f
                            audit["outgoing_ez_sum"] += total_f * z_out
                            audit["loss_sum"] += loss
                            audit["energy_loss_sum"] += total_i * loss
                            audit["routing_loss_sum"] += routing_loss
                            audit["drift_energy_sum"] += (
                                (2.0 / 3.0) * (etr_f - etr_i)
                                - theta * (erot_f - erot_i))
                            expected_etr_f = total_f * expected_z_out
                            expected_erot_f = total_f * (1.0 - expected_z_out)
                            audit["expected_drift_energy_sum"] += (
                                (2.0 / 3.0) * (expected_etr_f - etr_i)
                                - theta * (expected_erot_f - erot_i))
                            audit["theta_energy_sum"] += total_i * theta
                    if added:
                        ntc_collision_pairs += 1
                        if (p1 in step_collision_particles
                                or p2 in step_collision_particles):
                            ntc_repeated_particle_pairs += 1
                        step_collision_particles.update((p1, p2))
                    if pressure_accumulator is not None:
                        accumulate_pij_c(pressure_accumulator, v1, v2,
                                         state.velocity[p1], params.mass, speed,
                                         eij_override=normal)
            if hcs_rescale:
                current_ttr, _, current_total = state.temperatures(params.mass)
                mean_velocity = np.mean(state.velocity, axis=0)
                bulk_temperature = (params.mass * float(np.dot(
                    mean_velocity, mean_velocity)) / 3.0)
                maximum_bulk_to_thermal_temperature_ratio = max(
                    maximum_bulk_to_thermal_temperature_ratio,
                    bulk_temperature / max(current_ttr, 1.0e-30))
                current_reference = current_ttr if sphere else current_total
                if current_reference <= 0.0:
                    raise FloatingPointError(
                        "cannot rescale an HCS state with non-positive temperature")
                scale = math.sqrt(rescale_reference / current_reference)
                state.rescale_thermal_state(scale)
                # The rescale exactly undoes this step's collisional cooling,
                # so the relative-speed distribution is stationary and the
                # majorant must NOT be multiplied by scale: doing so ratchets
                # it up by the cooling factor every step, exp(0.077(1-a^2)tau)
                # in total, which is what OOM-killed every alpha<=0.9 task.
                # Only this step's observed maximum, carried to the rescaled
                # velocities, may raise it.
                vrmax = max(vrmax, vrmax_temp * scale)
                thermal = math.sqrt(2.0 * current_reference * scale * scale
                                    / params.mass)
                if vrmax > majorant_runaway_ratio * thermal:
                    raise RuntimeError(
                        f"NTC majorant runaway: vrmax={vrmax:.4g} is "
                        f"{vrmax / thermal:.1f} thermal speeds at tau={tau:.1f}")
            elif vrmax < vrmax_temp:
                vrmax = vrmax_temp
            state.advance_axes(dt)
            time += dt
    non_gaussian_summary = non_gaussian.close()
    total_seconds = wallclock.perf_counter() - march_started
    closure_collision_seconds = 0.0 if kernel is None else kernel.closure_seconds
    closure_seconds = closure_state_seconds + closure_collision_seconds
    diagnostics = {
        "particles": count, "collisions": collisions,
        "cpp": collisions / float(count), "sigma_c": params.sigma_c,
        "volume": volume, "number_density": count / volume,
        "routing": routing, "angular": angular, "flow": flow_mode,
        "minimum_Ftr": None if not np.isfinite(minimum_ftr) else minimum_ftr,
        "maximum_Ftr": None if not np.isfinite(maximum_ftr) else maximum_ftr,
        "negative_energy_repairs": 0 if kernel is None else kernel.negative_energy_repairs,
        "closure_seconds": closure_seconds,
        "closure_state_seconds": closure_state_seconds,
        "closure_collision_seconds": closure_collision_seconds,
        "closure_state_updates": closure_state_updates,
        "closure_state_update_cpp": closure_state_update_cpp,
        "runtime_seconds": total_seconds,
        # Only state reduction/interpolation is auxiliary closure overhead.
        # The per-collision variational draw is the collision model itself and
        # is reported separately instead of being mislabelled as overhead.
        "closure_overhead_fraction": (
            closure_state_seconds / max(total_seconds, 1.0e-30)),
        "closure_total_fraction": closure_seconds / max(total_seconds, 1.0e-30),
        "out_of_domain_fraction": (0.0 if not isinstance(closure, VariationalClosure)
                                    else closure.out_of_domain_fraction),
        "correction_fallback_policy": "base_law",
        "correction_fallback_queries": correction_fallback_queries,
        "correction_fallback_fraction": (
            correction_fallback_queries / max(closure_state_updates, 1)),
        "evaluation_start_tau": evaluation_start_tau,
        "evaluation_closure_queries": evaluation_closure_queries,
        "evaluation_correction_fallback_queries": (
            evaluation_correction_fallback_queries),
        "correction_fallback_fraction_in_evaluation_window": (
            evaluation_correction_fallback_queries
            / max(evaluation_closure_queries, 1)),
        "out_of_domain_fraction_by_feature": (
            {} if not isinstance(closure, VariationalClosure) else dict(zip(
                FEATURE_NAMES,
                (closure.out_of_domain_by_feature
                 / max(closure.total_queries, 1)).tolist()))),
        "sampling_excursion_fraction_by_feature": (
            {} if not isinstance(closure, VariationalClosure) else dict(zip(
                FEATURE_NAMES,
                (closure.sampling_excursion_by_feature
                 / max(closure.total_queries, 1)).tolist()))),
        "energy_axis_clamps": (0 if not isinstance(closure, VariationalClosure)
                                else closure.energy_axis_clamps),
        "energy_axis_clamp_fraction": (
            0.0 if not isinstance(closure, VariationalClosure)
            else closure.energy_axis_clamps / max(collisions, 1)),
        "energy_monotonic_repairs": (
            0 if not isinstance(closure, VariationalClosure)
            else closure.energy_monotonic_repairs),
        "maximum_energy_monotonic_repair": (
            0.0 if not isinstance(closure, VariationalClosure)
            else closure.maximum_energy_monotonic_repair),
        "energy_interpolation": (None if not isinstance(closure, VariationalClosure)
                                 else closure.energy_interpolation),
        "output": str(output_path),
        "pressure_output": None if pressure_path is None else str(pressure_path),
        "orientation_output": (None if orientation_path is None
                               else str(orientation_path)),
        "hcs_rescale_temperature": hcs_rescale,
        "minimum_theta_tr_over_rot": (
            None if not np.isfinite(theta_minimum) else theta_minimum),
        "maximum_theta_tr_over_rot": (
            None if not np.isfinite(theta_maximum) else theta_maximum),
        "minimum_theta_hull_log_margin": (
            None if not np.isfinite(minimum_theta_hull_log_margin)
            else minimum_theta_hull_log_margin),
        "closure_theta_guard_fraction": theta_guard_fraction,
        "maximum_bulk_to_thermal_temperature_ratio": (
            maximum_bulk_to_thermal_temperature_ratio),
        "final_center_of_mass_speed": float(np.linalg.norm(
            np.mean(state.velocity, axis=0))),
        "ntc": {
            "initial_vrmax": initial_vrmax, "final_vrmax": vrmax,
            "initial_vrmax_temperature_bound": vrmax_temperature_bound,
            "final_vrmax_over_initial": vrmax / initial_vrmax,
            "mean_candidates_per_step": ntc_candidates / max(ntc_steps, 1),
            "peak_candidates_per_step": peak_ntc_candidates,
            "candidate_ceiling_per_step": max_ntc_candidates,
            "acceptance_fraction": collisions / 2.0 / max(ntc_candidates, 1),
            "actual_collision_pairs": ntc_collision_pairs,
            "repeated_particle_pairs_same_step": ntc_repeated_particle_pairs,
            "repeated_particle_pair_fraction": (
                ntc_repeated_particle_pairs / max(ntc_collision_pairs, 1)),
            "majorant_violations": ntc_violations,
            # Candidate fraction measures how often the proposal majorant was
            # actually exceeded.  Keep the older accepted-pair normalisation
            # as a conservative diagnostic for compatibility with prior runs.
            "majorant_violation_fraction": ntc_violations / max(ntc_candidates, 1),
            "majorant_violations_per_accepted_pair": (
                ntc_violations / max(collisions / 2.0, 1.0)),
        },
        # Linux reports ru_maxrss in KiB.
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "exact_initial_temperatures": exact_initial_temperatures,
        "non_gaussian": non_gaussian_summary,
    }
    if audit_enabled:
        accepted = int(audit["accepted_pairs"])
        post_ntc = int(audit["post_ntc_pairs"])
        if accepted:
            diagnostics["collision_audit"] = {
                "post_ntc_pairs": post_ntc,
                "accepted_pairs": accepted,
                "orientation_acceptance": accepted / max(post_ntc, 1),
                "post_ntc_z_mean": audit["post_ntc_z_sum"] / max(post_ntc, 1),
                "post_ntc_z_energy_weighted": (
                    audit["post_ntc_ez_sum"] / audit["post_ntc_energy_sum"]),
                "accepted_z_mean": audit["accepted_z_sum"] / accepted,
                "accepted_z_energy_weighted": (
                    audit["accepted_ez_sum"] / audit["accepted_energy_sum"]),
                "outgoing_z_energy_weighted": (
                    audit["outgoing_ez_sum"] / audit["outgoing_energy_sum"]),
                "loss_mean": audit["loss_sum"] / accepted,
                "loss_energy_weighted": (
                    audit["energy_loss_sum"] / audit["accepted_energy_sum"]),
                "routing_loss_mean": audit["routing_loss_sum"] / accepted,
                "theta_energy_weighted": (
                    audit["theta_energy_sum"] / audit["accepted_energy_sum"]),
                "temperature_ratio_drift_per_pair_energy": (
                    audit["drift_energy_sum"] / audit["accepted_energy_sum"]),
                "expected_temperature_ratio_drift_per_pair_energy": (
                    audit["expected_drift_energy_sum"]
                    / audit["accepted_energy_sum"]),
                "cell_features": {
                    "samples": feature_count,
                    "mean": dict(zip(
                        FEATURE_NAMES,
                        (feature_sum / max(feature_count, 1)).tolist())),
                    "minimum": dict(zip(FEATURE_NAMES, feature_min.tolist())),
                    "maximum": dict(zip(FEATURE_NAMES, feature_max.tolist())),
                },
                "domain_features": {
                    "samples": feature_count,
                    "mean": dict(zip(
                        FEATURE_NAMES,
                        (domain_feature_sum / max(feature_count, 1)).tolist())),
                    "minimum": dict(zip(
                        FEATURE_NAMES, domain_feature_min.tolist())),
                    "maximum": dict(zip(
                        FEATURE_NAMES, domain_feature_max.tolist())),
                },
            }
        else:
            diagnostics["collision_audit"] = {
                "post_ntc_pairs": post_ntc, "accepted_pairs": 0}
    diagnostics["elastic_limit"] = elastic_limit
    diagnostics["runtime_gate"] = (runtime_gate_status(diagnostics)
                                   if routing in ("variational_v2", "elastic_bl")
                                   else None)
    return diagnostics
