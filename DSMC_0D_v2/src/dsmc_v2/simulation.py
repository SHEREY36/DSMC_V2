"""Conservative public HCS DSMC driver based on the proven v1 time march."""

from __future__ import annotations

import math
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
    if float(diagnostics["out_of_domain_fraction"]) >= 1.0e-3:
        reasons.append("out_of_domain_fraction_not_below_0.001")
    if float(diagnostics["closure_overhead_fraction"]) >= 0.05:
        reasons.append("closure_overhead_fraction_not_below_0.05")
    if int(diagnostics.get("energy_axis_clamps", 0)) != 0:
        reasons.append("energy_axis_clamps_not_zero")
    if int(diagnostics.get("energy_monotonic_repairs", 0)) != 0:
        reasons.append("energy_monotonic_repairs_not_zero")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "limits": {
            "negative_energy_repairs": 0,
            "out_of_domain_fraction_exclusive_maximum": 1.0e-3,
            "closure_overhead_fraction_exclusive_maximum": 0.05,
            "energy_axis_clamps": 0,
            "energy_monotonic_repairs": 0,
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
    routing, angular, closure = _closure_from_config(config)
    if sphere:
        models = kernel = None
    else:
        dissipation = config["preprocessing"]["dissipation"]
        model_root = config["preprocessing"].get("model_root", "models")
        if routing == "variational_v2":
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
    vrmax = 5.0 * np.sqrt(2.0) * np.sqrt(ktt / params.mass)
    time, collisions, output_index = 0.0, 0, 0
    workspace = NTCWorkspace(capacity=1024, seed=seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_ftr, minimum_ftr = -np.inf, np.inf
    closure_seconds = 0.0
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
          orientation_context as orientation_handle):
        while time < end_time and (tau_end is None or collisions / count < tau_end
                                   or collisions / count >= output_index * dtau):
            tau = collisions / float(count)
            if tau >= output_index * dtau:
                _write_row(handle, time, tau, state, params.mass)
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
            if kernel is not None and routing == "ctc_moment16":
                features = legacy_cell_features(state.velocity, state.omega, state.axis,
                                                params.mass, params.inertia, sphere=False)
                ftr = closure.routing_fraction(alpha, theta, params.aspect_ratio, features)
                kernel.set_cell_routing(ftr)
                minimum_ftr, maximum_ftr = min(minimum_ftr, ftr), max(maximum_ftr, ftr)
            elif kernel is not None and routing == "variational_v2":
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
                kernel.set_cell_variational(
                    closure.kernel_state(closure_alpha, theta, params.aspect_ratio,
                                         features, domain_features))
                closure_seconds += wallclock.perf_counter() - closure_started

            if routing == "variational_v2" and not getattr(kernel, "_enhanced", False):
                kernel.set_enhancement(
                    closure.xi_grid, float(np.max(closure.xi_enhancement)))
                kernel._enhanced = True
            inflation = (kernel.candidate_inflation
                         if (kernel is not None and routing == "variational_v2")
                         else 1.0)
            n_candidates = candidate_count(count, params.sigma_c * inflation,
                                           vrmax, volume, dt)
            vrmax_temp = 0.0
            if n_candidates > 0:
                vrmax_temp, accepted = workspace.screen_candidates(
                    state.velocity, count, n_candidates, vrmax)
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
                    if audit_enabled and not sphere:
                        pair_before = _pair_modal_energies(
                            state, p1, p2, v1, v2, params.mass)
                        pair_total = pair_before[0] + pair_before[1]
                        if pair_total > 0.0:
                            pair_z = pair_before[0] / pair_total
                            audit["post_ntc_pairs"] += 1
                            audit["post_ntc_z_sum"] += pair_z
                            audit["post_ntc_energy_sum"] += pair_total
                            audit["post_ntc_ez_sum"] += pair_total * pair_z
                    if sphere:
                        collisions += _sphere_collision(state, p1, p2, normal, v1, v2, cr, alpha)
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
                    if pressure_accumulator is not None:
                        accumulate_pij_c(pressure_accumulator, v1, v2,
                                         state.velocity[p1], params.mass, speed,
                                         eij_override=normal)
            if vrmax < vrmax_temp:
                vrmax = vrmax_temp
            if hcs_rescale:
                current_ttr, _, current_total = state.temperatures(params.mass)
                current_reference = current_ttr if sphere else current_total
                if current_reference <= 0.0:
                    raise FloatingPointError(
                        "cannot rescale an HCS state with non-positive temperature")
                scale = math.sqrt(rescale_reference / current_reference)
                state.rescale_thermal_state(scale)
                # Every relative speed, including the current NTC majorant,
                # receives exactly the same similarity factor.
                vrmax *= scale
            state.advance_axes(dt)
            time += dt
    non_gaussian_summary = non_gaussian.close()
    total_seconds = wallclock.perf_counter() - march_started
    closure_seconds += 0.0 if kernel is None else kernel.closure_seconds
    diagnostics = {
        "particles": count, "collisions": collisions,
        "cpp": collisions / float(count), "sigma_c": params.sigma_c,
        "volume": volume, "number_density": count / volume,
        "routing": routing, "angular": angular, "flow": flow_mode,
        "minimum_Ftr": None if not np.isfinite(minimum_ftr) else minimum_ftr,
        "maximum_Ftr": None if not np.isfinite(maximum_ftr) else maximum_ftr,
        "negative_energy_repairs": 0 if kernel is None else kernel.negative_energy_repairs,
        "closure_seconds": closure_seconds,
        "runtime_seconds": total_seconds,
        "closure_overhead_fraction": closure_seconds / max(total_seconds, 1.0e-30),
        "out_of_domain_fraction": (0.0 if not isinstance(closure, VariationalClosure)
                                    else closure.out_of_domain_fraction),
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
    diagnostics["runtime_gate"] = (runtime_gate_status(diagnostics)
                                   if routing == "variational_v2" else None)
    return diagnostics
