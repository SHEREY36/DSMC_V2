"""The v1 BL/GMM collision kernel with only routing and angle extension points."""

from __future__ import annotations

import numpy as np

from .angular import sample_direction


def prepare_theta(theta: float) -> float:
    return float(np.clip(round(theta * 10.0) / 10.0, 0.1, 1.2))


def rotational_collision_number(theta: float, alpha: float) -> float:
    return 1.67 if alpha >= 1.0 else 0.39 * theta**2 + 0.09 * theta + 1.67


def rank0_ftr(c_alpha: float, theta: float) -> float:
    return float(c_alpha) * 3.0 * theta / (3.0 * theta + 2.0)


def chi_hs(mu: float, alpha: float) -> float:
    denominator = np.sqrt(max(1.0 - (1.0 - alpha * alpha) * mu * mu, 1.0e-30))
    cosine = (1.0 - (1.0 + alpha) * mu * mu) / denominator
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _perpendicular(ghat: np.ndarray) -> np.ndarray:
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, ghat)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    result = trial - np.dot(trial, ghat) * ghat
    return result / np.linalg.norm(result)


def legacy_scatter(vrel: np.ndarray, normal: np.ndarray, chi: float,
                   magnitude: float, epsilon: float) -> np.ndarray:
    ghat = vrel / max(np.linalg.norm(vrel), 1.0e-30)
    mu = float(np.clip(np.dot(normal, ghat), 0.0, 1.0))
    tangent = normal - mu * ghat
    tangent = _perpendicular(ghat) if np.linalg.norm(tangent) <= 1.0e-12 \
        else tangent / np.linalg.norm(tangent)
    second = np.cross(ghat, tangent)
    post = (np.cos(chi) * ghat - np.sin(chi)
            * (np.cos(epsilon) * tangent - np.sin(epsilon) * second))
    return magnitude * post / np.linalg.norm(post)


class SpherocylinderKernel:
    def __init__(self, params, models, alpha: float, beta_a: float, beta_b: float,
                 c_alpha: float, closure, routing_mode: str, angular_mode: str,
                 direction_rng: np.random.Generator, vss_rng: np.random.Generator,
                 equilibration_time: float = 0.0,
                 isotropic_epsilon: bool = True):
        self.params, self.models, self.alpha = params, models, float(alpha)
        self.beta_a, self.beta_b, self.c_alpha = beta_a, beta_b, c_alpha
        self.closure, self.routing_mode, self.angular_mode = closure, routing_mode, angular_mode
        self.direction_rng, self.vss_rng = direction_rng, vss_rng
        self.equilibration_time = float(equilibration_time)
        self.isotropic_epsilon = bool(isotropic_epsilon)
        self.loss = models.loss_parameters(alpha, params.aspect_ratio) if alpha < 1.0 else {
            "gamma_max": 0.0, "one_hit_probability": 1.0}
        self.cell_routing = None

    def set_cell_routing(self, value: float | None) -> None:
        self.cell_routing = value

    def collide(self, state, p1: int, p2: int, normal: np.ndarray,
                v1: np.ndarray, v2: np.ndarray, vrel: np.ndarray,
                relative_speed: float, time: float, theta: float) -> int:
        vcom = 0.5 * (v1 + v2)
        v1com, v2com = v1 - vcom, v2 - vcom
        etr_i = 0.5 * self.params.mass * (np.dot(v1com, v1com) + np.dot(v2com, v2com))
        erot_i = state.rotational_energy[p1] + state.rotational_energy[p2]
        total_i = etr_i + erot_i
        if total_i <= 0.0:
            return 0
        eps_tr_i = etr_i / total_i
        eps_r1_i = state.rotational_energy[p1] / erot_i if erot_i > 0.0 else 0.5
        in_equilibration = time < self.equilibration_time
        pr = min(1.0 / rotational_collision_number(theta, self.alpha), 0.5)
        draw = np.random.random()
        relax1, relax2 = draw < pr, draw >= pr and draw < 2.0 * pr
        if relax1 or relax2:
            if self.alpha >= 1.0:
                eps_tr_f, sampled = np.random.beta(2.0, 2.0), np.random.random()
            else:
                sampled_pair = self.models.cond_gmm.sample_conditionals(
                    prepare_theta(theta), eps_tr_i, eps_r1_i, 1)[0]
                eps_tr_f, sampled = sampled_pair
            eps_r1_f = sampled if relax1 else 1.0 - sampled
        else:
            eps_tr_f, eps_r1_f = eps_tr_i, eps_r1_i
        gamma = 0.0 if in_equilibration or self.alpha >= 1.0 else (
            np.random.beta(self.beta_a, self.beta_b) * self.loss["gamma_max"]
            * self.loss["one_hit_probability"])
        delta = gamma * total_i
        if self.routing_mode == "legacy_rank0":
            ftr = rank0_ftr(self.c_alpha, theta)
        else:
            if self.cell_routing is None:
                raise RuntimeError("moment16 routing was not evaluated for the current cell")
            ftr = self.cell_routing
        etr_f = eps_tr_f * total_i - ftr * delta
        erot_f = (1.0 - eps_tr_f) * total_i - (1.0 - ftr) * delta
        # Preserve the v1 reservoir handling exactly.
        if etr_f < 0.0:
            erot_f += etr_f
            etr_f = 1.0e-30
        if erot_f < 0.0:
            etr_f += erot_f
            erot_f = 1.0e-30
        state.rotational_energy[p1] = eps_r1_f * erot_f
        state.rotational_energy[p2] = (1.0 - eps_r1_f) * erot_f

        ghat = vrel / max(relative_speed, 1.0e-30)
        magnitude = 2.0 * max(np.sqrt(etr_f / self.params.mass), 1.0e-14)
        if self.angular_mode == "legacy" or in_equilibration:
            mu = abs(float(np.dot(normal, ghat)))
            chi = chi_hs(mu, 1.0 if in_equilibration else self.alpha)
            epsilon = np.random.uniform(0.0, 2.0 * np.pi) if self.isotropic_epsilon else 0.0
            gpost = legacy_scatter(vrel, normal, chi, magnitude, epsilon)
        else:
            direction = sample_direction(ghat,
                self.closure.alpha_eff(self.alpha, self.params.aspect_ratio), self.vss_rng)
            gpost = magnitude * direction
        state.velocity[p1] = vcom + 0.5 * gpost
        state.velocity[p2] = vcom - 0.5 * gpost

        if self.closure is not None:
            directions = self.closure.spin_directions(
                self.alpha, theta, self.params.aspect_ratio, v1, v2,
                state.omega[p1], state.omega[p2], state.axis[p1], state.axis[p2],
                np.array([eps_tr_f, eps_r1_f]), self.params.mass,
                self.params.inertia, self.direction_rng)
            state.set_spin_directions((p1, p2), directions)
        else:
            state.preserve_spin_directions((p1, p2))
        return 2
