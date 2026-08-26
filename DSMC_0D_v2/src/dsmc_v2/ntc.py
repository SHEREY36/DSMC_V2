"""Unbiased no-time-counter proposal clock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_distinct_pair(count: int, rng: np.random.Generator) -> tuple[int, int]:
    if count < 2:
        raise ValueError("at least two particles are required")
    first = int(rng.integers(count))
    second = int(rng.integers(count - 1))
    if second >= first:
        second += 1
    return (first, second) if first < second else (second, first)


@dataclass
class NTCClock:
    carry: float = 0.0

    def candidate_count(self, particle_count: int, volume: float, dt: float,
                        rate_majorant: float) -> int:
        if volume <= 0.0 or dt < 0.0 or rate_majorant < 0.0:
            raise ValueError("invalid NTC clock argument")
        expected = (particle_count * (particle_count - 1) / (2.0 * volume)
                    * rate_majorant * dt + self.carry)
        result = int(np.floor(expected))
        self.carry = expected - result
        return result


def acceptance_probability(relative_speed: float, cross_section: float,
                           speed_majorant: float, area_majorant: float) -> float:
    if speed_majorant <= 0.0 or area_majorant <= 0.0:
        return 0.0
    probability = relative_speed * cross_section / (speed_majorant * area_majorant)
    if probability > 1.0 + 1.0e-12:
        raise RuntimeError(f"NTC majorant violated: acceptance={probability}")
    return float(np.clip(probability, 0.0, 1.0))

