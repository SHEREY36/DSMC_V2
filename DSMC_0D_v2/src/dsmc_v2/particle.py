"""The proven v1 particle geometry and frozen collision cross-section."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParticleParameters:
    aspect_ratio: float
    radius: float
    mass: float
    diameter: float
    cylinder_length: float
    volume: float
    inertia: float
    sigma_c: float

    @property
    def inverse_mass(self) -> float:
        return 1.0 / self.mass


def frozen_sigma_c(aspect_ratio: float, diameter: float) -> float:
    ar = float(aspect_ratio)
    return float(np.pi * diameter**2 * (0.32 * ar**2 + 0.694 * ar - 0.0213))


def particle_parameters(config: dict) -> ParticleParameters:
    particle = config["particle"]
    ar = float(particle["AR"])
    radius = float(particle["radius"])
    mass = float(particle["mass"])
    diameter = 2.0 * radius
    length = (ar - 1.0) * diameter
    volume = np.pi * diameter**3 / 6.0 + np.pi * length * radius**2
    rho = mass / volume
    inertia = (
        np.pi * rho * diameter**2 * length**3 / 48.0
        + 3.0 * np.pi * rho * diameter**4 * length / 64.0
        + np.pi * rho * diameter**5 / 60.0
        + np.pi * rho * diameter**3 * length**2 / 24.0
    )
    sphere = bool(config.get("simulation", {}).get("sphere_collision", False))
    sigma = np.pi * diameter**2 if sphere else frozen_sigma_c(ar, diameter)
    return ParticleParameters(ar, radius, mass, diameter, length, volume,
                              inertia, sigma)
