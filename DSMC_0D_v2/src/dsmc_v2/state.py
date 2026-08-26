"""Scalar v1 state plus passive axisymmetric vector bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleState:
    velocity: np.ndarray
    rotational_energy: np.ndarray
    omega: np.ndarray
    axis: np.ndarray
    inertia: float

    @property
    def count(self) -> int:
        return len(self.velocity)

    def normalize_constraints(self) -> None:
        self.axis /= np.linalg.norm(self.axis, axis=1)[:, None]
        self.omega -= np.einsum("ni,ni->n", self.omega, self.axis)[:, None] * self.axis
        expected = 0.5 * self.inertia * np.einsum("ni,ni->n", self.omega, self.omega)
        if not np.allclose(expected, self.rotational_energy, rtol=2.0e-12, atol=2.0e-14):
            raise ValueError("omega and scalar rotational energy are inconsistent")

    def set_spin_directions(self, indices: tuple[int, int], directions: np.ndarray) -> None:
        for index, direction in zip(indices, np.asarray(directions)):
            tangent = direction - np.dot(direction, self.axis[index]) * self.axis[index]
            norm = np.linalg.norm(tangent)
            if norm <= 1.0e-12:
                raise ValueError("degenerate tangent spin direction")
            magnitude = np.sqrt(2.0 * max(self.rotational_energy[index], 0.0) / self.inertia)
            self.omega[index] = magnitude * tangent / norm

    def preserve_spin_directions(self, indices: tuple[int, int]) -> None:
        directions = []
        for index in indices:
            norm = np.linalg.norm(self.omega[index])
            if norm <= 1.0e-14:
                trial = np.cross(self.axis[index], np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(trial) <= 1.0e-12:
                    trial = np.cross(self.axis[index], np.array([0.0, 1.0, 0.0]))
                directions.append(trial)
            else:
                directions.append(self.omega[index] / norm)
        self.set_spin_directions(indices, np.asarray(directions))

    def advance_axes(self, dt: float) -> None:
        """Rodrigues update for du/dt=omega cross u; no scalar state changes."""
        for i in range(self.count):
            speed = np.linalg.norm(self.omega[i])
            angle = speed * float(dt)
            if angle <= 1.0e-14:
                continue
            direction = self.omega[i] / speed
            axis = self.axis[i]
            self.axis[i] = (axis * np.cos(angle)
                            + np.cross(direction, axis) * np.sin(angle)
                            + direction * np.dot(direction, axis) * (1.0 - np.cos(angle)))
        self.axis /= np.linalg.norm(self.axis, axis=1)[:, None]

    def temperatures(self, mass: float) -> tuple[float, float, float]:
        n = self.count
        ttr = mass * np.sum(self.velocity**2) / (3.0 * n)
        trot = np.sum(self.rotational_energy) / n
        return float(ttr), float(trot), float((3.0 * ttr + 2.0 * trot) / 5.0)


def initialize_particles(count: int, ttr: float, trot: float, mass: float,
                         inertia: float, closure_rng: np.random.Generator,
                         sphere: bool = False) -> ParticleState:
    """Use the exact v1 global RNG draws, then add axes on an isolated stream."""
    velocity = np.random.randn(count, 3) * np.sqrt(ttr / mass)
    omega = np.random.randn(count, 3) * np.sqrt(trot / inertia)
    omega[:, 0] = 0.0
    rotational_energy = 0.5 * inertia * (omega[:, 1]**2 + omega[:, 2]**2)
    velocity -= np.sum(velocity, axis=0) / count
    axis = closure_rng.normal(size=(count, 3))
    for i in range(count):
        norm_w = np.linalg.norm(omega[i])
        if norm_w > 1.0e-14:
            what = omega[i] / norm_w
            axis[i] -= np.dot(axis[i], what) * what
        if np.linalg.norm(axis[i]) <= 1.0e-12:
            axis[i] = np.cross(omega[i], np.array([1.0, 0.0, 0.0]))
        axis[i] /= np.linalg.norm(axis[i])
    if sphere:
        omega[:] = 0.0
        rotational_energy[:] = 0.0
    return ParticleState(velocity, rotational_energy, omega, axis, inertia)

