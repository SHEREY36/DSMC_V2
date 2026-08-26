"""Particle state and axisymmetric free-flight evolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleState:
    velocity: np.ndarray
    omega: np.ndarray
    axis: np.ndarray

    def __post_init__(self) -> None:
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.omega = np.asarray(self.omega, dtype=float)
        self.axis = np.asarray(self.axis, dtype=float)
        if self.velocity.shape != self.omega.shape or self.velocity.shape != self.axis.shape \
                or self.velocity.ndim != 2 or self.velocity.shape[1] != 3:
            raise ValueError("velocity, omega and axis must have shape (N,3)")
        self.normalize_constraints()

    @property
    def count(self) -> int:
        return len(self.velocity)

    def normalize_constraints(self) -> None:
        norm = np.linalg.norm(self.axis, axis=1)
        if np.any(norm <= 1.0e-14):
            raise ValueError("particle axes must be nonzero")
        self.axis /= norm[:, None]
        self.omega -= np.einsum("ni,ni->n", self.omega, self.axis)[:, None] * self.axis

    def advance_axes(self, dt: float) -> None:
        """Second-order Rodrigues update for du/dt=omega cross u."""
        angle = np.linalg.norm(self.omega, axis=1) * float(dt)
        for i, value in enumerate(angle):
            if value < 1.0e-14:
                self.axis[i] += dt * np.cross(self.omega[i], self.axis[i])
                continue
            direction = self.omega[i] / np.linalg.norm(self.omega[i])
            u = self.axis[i]
            self.axis[i] = (u * np.cos(value) + np.cross(direction, u) * np.sin(value)
                            + direction * np.dot(direction, u) * (1.0 - np.cos(value)))
        self.normalize_constraints()

    def temperatures(self, mass: float, moi_perpendicular: float) -> tuple[float, float]:
        peculiar = self.velocity - np.mean(self.velocity, axis=0)
        ttr = mass * np.einsum("ni,ni->", peculiar, peculiar) / (3.0 * self.count)
        trot = moi_perpendicular * np.einsum("ni,ni->", self.omega, self.omega) / (2.0 * self.count)
        return float(ttr), float(trot)

