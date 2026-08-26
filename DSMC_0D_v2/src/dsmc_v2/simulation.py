"""Conservative public HCS DSMC driver based on the proven v1 time march."""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import cell_features

from .artifact import MicroscopicClosure
from .kernel import SpherocylinderKernel
from .legacy_models import LegacyModels
from .ntc import NTCWorkspace, candidate_count
from .particle import particle_parameters
from .pressure import accumulate_pij_c, compute_pij_k, normalise_pij_c
from .state import initialize_particles


def _closure_from_config(config: dict) -> tuple[str, str, MicroscopicClosure | None]:
    closure_config = config.get("microscopic_closure", {})
    routing = closure_config.get("routing", "legacy_rank0")
    angular = closure_config.get("angular", "legacy")
    if routing not in ("legacy_rank0", "ctc_moment16"):
        raise ValueError("routing must be legacy_rank0 or ctc_moment16")
    if angular not in ("legacy", "ctc_vss_rank2"):
        raise ValueError("angular must be legacy or ctc_vss_rank2")
    if routing == "legacy_rank0" and angular == "legacy":
        return routing, angular, None
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


def run_simulation(config: dict, seed: int, output_path: str | Path,
                   pressure_path: str | Path | None = None) -> dict:
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
                                 axis_rng, sphere)
    routing, angular, closure = _closure_from_config(config)
    if sphere:
        models = kernel = None
    else:
        dissipation = config["preprocessing"]["dissipation"]
        models = LegacyModels(config["preprocessing"].get("model_root", "models"),
                              params.aspect_ratio, float(dissipation["beta_a"]),
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
    vrmax = 5.0 * np.sqrt(2.0) * np.sqrt(ktt / params.mass)
    time, collisions, output_index = 0.0, 0, 0
    workspace = NTCWorkspace(capacity=1024, seed=seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_ftr, minimum_ftr = -np.inf, np.inf
    pressure_path = Path(pressure_path) if pressure_path is not None else None
    if flow_mode == "usf" and pressure_path is None:
        pressure_path = output_path.with_name(output_path.stem + "_pressure.txt")
    if pressure_path is not None:
        pressure_path.parent.mkdir(parents=True, exist_ok=True)
    pressure_accumulator = np.zeros((3, 3)) if flow_mode == "usf" else None
    last_pressure_time = 0.0
    pressure_context = pressure_path.open("w", buffering=65536) if pressure_path else nullcontext(None)
    with output_path.open("w", buffering=65536) as handle, pressure_context as pressure_handle:
        while time < end_time and (tau_end is None or collisions / count < tau_end
                                   or collisions / count >= output_index * dtau):
            tau = collisions / float(count)
            if tau >= output_index * dtau:
                _write_row(handle, time, tau, state, params.mass)
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
                output_index += 1
            if flow_mode == "usf":
                state.velocity[:, 0] -= shear_rate * state.velocity[:, 1] * dt
            ttr, trot, _ = state.temperatures(params.mass)
            theta = ttr / trot if trot > 0.0 else 1.0
            if kernel is not None and routing == "ctc_moment16":
                features = cell_features(state.velocity, state.omega, state.axis,
                                         params.mass, params.inertia, sphere=False)
                ftr = closure.routing_fraction(alpha, theta, params.aspect_ratio, features)
                kernel.set_cell_routing(ftr)
                minimum_ftr, maximum_ftr = min(minimum_ftr, ftr), max(maximum_ftr, ftr)

            n_candidates = candidate_count(count, params.sigma_c, vrmax, volume, dt)
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
                    if sphere:
                        collisions += _sphere_collision(state, p1, p2, normal, v1, v2, cr, alpha)
                    else:
                        collisions += kernel.collide(
                            state, p1, p2, normal, v1, v2, vrel, speed, time, theta)
                    if pressure_accumulator is not None:
                        accumulate_pij_c(pressure_accumulator, v1, v2,
                                         state.velocity[p1], params.mass, speed,
                                         eij_override=normal)
            if vrmax < vrmax_temp:
                vrmax = vrmax_temp
            state.advance_axes(dt)
            time += dt
    return {
        "particles": count, "collisions": collisions,
        "cpp": collisions / float(count), "sigma_c": params.sigma_c,
        "routing": routing, "angular": angular, "flow": flow_mode,
        "minimum_Ftr": None if not np.isfinite(minimum_ftr) else minimum_ftr,
        "maximum_Ftr": None if not np.isfinite(maximum_ftr) else maximum_ftr,
        "output": str(output_path),
        "pressure_output": None if pressure_path is None else str(pressure_path),
    }
