"""Sequential-state 0D DSMC using the versioned CTC collision operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsmc_v2_contracts import cell_features

from .angular import sample_direction
from .artifact import CollisionArtifactV2
from .ntc import NTCClock, acceptance_probability, sample_distinct_pair
from .reconstruction import reconstruct_post_state
from .state import ParticleState


@dataclass
class StepDiagnostics:
    candidates: int = 0
    accepted: int = 0
    reconstruction_draws: int = 0
    infeasible_draws: int = 0
    maximum_energy_error: float = 0.0
    maximum_angular_momentum_error: float = 0.0


class HomogeneousDSMC:
    def __init__(self, state: ParticleState, artifact: CollisionArtifactV2,
                 alpha: float, aspect_ratio: float, mass: float = 1.0,
                 moi_perpendicular: float = 1.0, mode: str = "pair_resolved",
                 seed: int = 12345, max_reconstruction_draws: int = 64):
        if mode not in ("pair_resolved", "moment16"):
            raise ValueError("mode must be pair_resolved or moment16")
        if not 0.0 < alpha <= 1.0 or aspect_ratio < 1.0:
            raise ValueError("invalid alpha or aspect ratio")
        self.state, self.artifact = state, artifact
        self.alpha, self.aspect_ratio = float(alpha), float(aspect_ratio)
        self.mass, self.moi = float(mass), float(moi_perpendicular)
        self.mode, self.max_reconstruction_draws = mode, int(max_reconstruction_draws)
        self.rng, self.clock = np.random.default_rng(seed), NTCClock()

    def _total_internal_energy(self) -> float:
        peculiar = self.state.velocity - np.mean(self.state.velocity, axis=0)
        return float(self.mass * np.einsum("ni,ni->", peculiar, peculiar)
                     + self.moi * np.einsum("ni,ni->", self.state.omega, self.state.omega))

    def _moment_outcome(self, first: int, second: int, gamma: float,
                        ftr: float, theta: float) -> np.ndarray | None:
        s = self.state
        template = self.artifact.sample_pair_outcome(
            s.velocity[first], s.velocity[second], s.omega[first], s.omega[second],
            s.axis[first], s.axis[second], self.alpha, theta, self.aspect_ratio, self.rng)
        g = s.velocity[second] - s.velocity[first]
        etr = 0.5 * self.mass * np.dot(g, g)
        er1 = self.moi * np.dot(s.omega[first], s.omega[first])
        er2 = self.moi * np.dot(s.omega[second], s.omega[second])
        total = etr + er1 + er2
        loss = gamma * total
        etr_post = etr - ftr * loss
        erot_post = er1 + er2 - (1.0 - ftr) * loss
        if etr_post < 0.0 or erot_post < 0.0:
            return None
        retained = etr_post + erot_post
        split = template[2] / max(template[2] + template[3], 1.0e-30)
        result = template.copy()
        result[0] = retained / max(total, 1.0e-30)
        result[1:4] = np.array([etr_post, split * erot_post,
                                (1.0 - split) * erot_post]) / max(retained, 1.0e-30)
        return result

    def step(self, dt: float, volume: float, shear_rate: float = 0.0) -> StepDiagnostics:
        state = self.state
        if shear_rate:
            state.velocity[:, 0] -= float(shear_rate) * state.velocity[:, 1] * float(dt)
        state.advance_axes(dt)
        ttr, trot = state.temperatures(self.mass, self.moi)
        theta = ttr / trot
        area = self.artifact.proposal_area(self.aspect_ratio)
        # Total energy is non-increasing through collisions, so this remains a
        # finite-population bound even if rotation transfers into translation.
        speed_majorant = 2.0 * np.sqrt(max(self._total_internal_energy(), 0.0) / self.mass)
        diagnostics = StepDiagnostics()
        diagnostics.candidates = self.clock.candidate_count(
            state.count, volume, dt, speed_majorant * area)
        if diagnostics.candidates == 0 or speed_majorant == 0.0:
            return diagnostics
        if self.mode == "moment16":
            features = cell_features(state.velocity, state.omega, state.axis,
                                     self.mass, self.moi, np.isclose(self.aspect_ratio, 1.0))
            sigma_cell, gamma_cell, ftr_cell = self.artifact.reduced_means(
                self.alpha, theta, self.aspect_ratio, features)
        for _ in range(diagnostics.candidates):
            first, second = sample_distinct_pair(state.count, self.rng)
            g = state.velocity[second] - state.velocity[first]
            relative_speed = float(np.linalg.norm(g))
            if self.mode == "pair_resolved":
                sigma = self.artifact.pair_cross_section(
                    state.velocity[first], state.velocity[second],
                    state.omega[first], state.omega[second],
                    state.axis[first], state.axis[second], self.aspect_ratio)
            else:
                sigma = sigma_cell
            probability = acceptance_probability(relative_speed, sigma, speed_majorant, area)
            if self.rng.random() >= probability or relative_speed <= 1.0e-30:
                continue
            accepted_result = None
            for _draw in range(self.max_reconstruction_draws):
                diagnostics.reconstruction_draws += 1
                if self.mode == "pair_resolved":
                    outcome = self.artifact.sample_pair_outcome(
                        state.velocity[first], state.velocity[second],
                        state.omega[first], state.omega[second],
                        state.axis[first], state.axis[second], self.alpha, theta,
                        self.aspect_ratio, self.rng)
                else:
                    outcome = self._moment_outcome(first, second, gamma_cell, ftr_cell, theta)
                    if outcome is None:
                        diagnostics.infeasible_draws += 1
                        continue
                alpha_eff = self.artifact.alpha_eff(self.alpha, self.aspect_ratio)
                direction = sample_direction(g / relative_speed, alpha_eff, self.rng)
                accepted_result = reconstruct_post_state(
                    state.velocity[first], state.velocity[second],
                    state.omega[first], state.omega[second],
                    state.axis[first], state.axis[second], direction, outcome,
                    self.mass, self.moi, self.aspect_ratio, self.rng)
                if accepted_result is not None:
                    break
                diagnostics.infeasible_draws += 1
            if accepted_result is None:
                continue
            state.velocity[first], state.velocity[second] = accepted_result.velocity1, accepted_result.velocity2
            state.omega[first], state.omega[second] = accepted_result.omega1, accepted_result.omega2
            diagnostics.accepted += 1
            diagnostics.maximum_energy_error = max(diagnostics.maximum_energy_error,
                                                     accepted_result.energy_error)
            diagnostics.maximum_angular_momentum_error = max(
                diagnostics.maximum_angular_momentum_error,
                accepted_result.angular_momentum_error)
        state.normalize_constraints()
        return diagnostics

