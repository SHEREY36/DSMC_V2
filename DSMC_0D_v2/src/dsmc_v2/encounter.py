"""Encounter-unit collision clock for the variational closure.

The exchange and angular kernels are fitted on complete CTC encounters: one
record per pair approach, from first contact to final separation, however many
contacts that approach contains.  A DSMC event must therefore occur at the
encounter rate, measured directly from the CTC shards as

    sigma(theta, AR) = (2 b_max)^2 N_hit / N_att .

The encounter cross-section is pure kinematics -- no force acts before first
contact -- so it depends on the translational/rotational temperature ratio and
the shape only, and is identical across restitution coefficients.

Schema v2 (current) stores sigma per node and its dynamic factor
D = sigma / A_bar, with A_bar = pi (d^2 + d L + L^2/8) the exact isotropic mean
projected excluded area.  The runtime interpolates D (linear in log theta,
flat outside the measured range, then linear in AR) and multiplies the exact
geometry back, so no fitted AR polynomial is involved.  Schema v1 (the ratio
to the v1 polynomial sigma_c) is still read for reproducing earlier campaigns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def mean_projected_area(aspect_ratio: float, diameter: float) -> float:
    """Isotropic mean projected excluded area of two spherocylinders."""
    length = (float(aspect_ratio) - 1.0) * diameter
    return math.pi * (diameter**2 + diameter * length + length**2 / 8.0)


def polynomial_sigma_c(aspect_ratio: float, diameter: float) -> float:
    ar = float(aspect_ratio)
    return math.pi * diameter**2 * (0.32 * ar**2 + 0.694 * ar - 0.0213)


class EncounterClock:
    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text())
        self.schema = data.get("schema")
        if self.schema == "encounter-cross-section-v2":
            rows, field = data["cross_section"], "dynamic_factor"
        elif self.schema == "encounter-cross-section-v1":
            rows, field = data["ratio"], "sigma_enc_over_sigma_c"
        else:
            raise ValueError(f"{path} is not an encounter cross-section table")
        self.aspect_ratios = np.array(sorted(float(key) for key in rows))
        self.tables = []
        for ar in self.aspect_ratios:
            row = rows[f"{ar:.4f}"]
            theta = np.asarray(row["theta"], dtype=float)
            value = np.asarray(row[field], dtype=float)
            order = np.argsort(theta)
            self.tables.append((np.log(theta[order]), value[order]))
        self.path = str(path)

    def _factor(self, theta: float, aspect_ratio: float) -> float:
        ar = float(aspect_ratio)
        if not self.aspect_ratios[0] - 1e-9 <= ar <= self.aspect_ratios[-1] + 1e-9:
            raise ValueError(f"AR={ar} outside the encounter table "
                             f"[{self.aspect_ratios[0]}, {self.aspect_ratios[-1]}]")
        log_theta = float(np.log(max(theta, 1.0e-12)))
        upper = int(np.searchsorted(self.aspect_ratios, ar).clip(1, len(self.aspect_ratios) - 1))
        lower = upper - 1
        span = self.aspect_ratios[upper] - self.aspect_ratios[lower]
        blend = 0.0 if span <= 0 else min(max((ar - self.aspect_ratios[lower]) / span, 0.0), 1.0)
        # Flat extrapolation in theta: the factor saturates at both ends
        # (translation- or rotation-dominated approach).
        at = [float(np.interp(log_theta, *self.tables[i])) for i in (lower, upper)]
        return (1.0 - blend) * at[0] + blend * at[1]

    def sigma(self, theta: float, aspect_ratio: float, diameter: float) -> float:
        """Encounter cross-section at the cell's temperature ratio."""
        factor = self._factor(theta, aspect_ratio)
        if self.schema == "encounter-cross-section-v2":
            return mean_projected_area(aspect_ratio, diameter) * factor
        return polynomial_sigma_c(aspect_ratio, diameter) * factor
