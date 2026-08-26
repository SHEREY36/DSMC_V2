"""Read-only description of the preserved v1 Borgnakke--Larsen loss law."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _parse_key(key: str) -> tuple[float, float]:
    alpha, aspect_ratio = key.strip()[1:-1].split(",")
    return float(alpha), float(aspect_ratio)


def _lookup(table: dict[str, float], alpha: float, aspect_ratio: float) -> float:
    """Match v1's alpha-then-AR linear interpolation inside its table hull."""
    points = np.array([(*_parse_key(key), float(value)) for key, value in table.items()])
    ars = np.unique(points[:, 1])
    if aspect_ratio < ars[0] - 1.0e-12 or aspect_ratio > ars[-1] + 1.0e-12:
        raise ValueError(f"AR={aspect_ratio} is outside the preserved BL table hull")
    per_ar = []
    for ar in ars:
        rows = points[np.isclose(points[:, 1], ar)]
        order = np.argsort(rows[:, 0])
        rows = rows[order]
        if alpha < rows[0, 0] - 1.0e-12 or alpha > rows[-1, 0] + 1.0e-12:
            raise ValueError(f"alpha={alpha} is outside the preserved BL table hull")
        per_ar.append(np.interp(alpha, rows[:, 0], rows[:, 2]))
    return float(np.interp(aspect_ratio, ars, per_ar))


@dataclass(frozen=True)
class LegacyBL:
    gamma_max: dict[str, float]
    one_hit: dict[str, float]
    beta_a: float = 1.21
    beta_b: float = 3.67

    @classmethod
    def load(cls, gamma_max_path: str | Path, one_hit_path: str | Path,
             beta_a: float = 1.21, beta_b: float = 3.67) -> "LegacyBL":
        return cls(json.loads(Path(gamma_max_path).read_text()),
                   json.loads(Path(one_hit_path).read_text()), beta_a, beta_b)

    def parameters(self, alpha: float, aspect_ratio: float) -> dict[str, float]:
        gamma_max = _lookup(self.gamma_max, alpha, aspect_ratio)
        one_hit = _lookup(self.one_hit, alpha, aspect_ratio)
        mean_beta = self.beta_a / (self.beta_a + self.beta_b)
        return {
            "gamma_max": gamma_max,
            "one_hit_probability": one_hit,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
            "mean_loss_fraction": gamma_max * one_hit * mean_beta,
        }
