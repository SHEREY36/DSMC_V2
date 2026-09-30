"""Encounter-unit collision clock for the variational closure.

The exchange and angular kernels are fitted on complete CTC encounters: one
record per pair approach, from first contact to final separation, however many
contacts that approach contains.  A DSMC event must therefore occur at the
encounter rate, not at the contact rate that the frozen v1 polynomial sigma_c
represents (sigma_c / sigma_enc = 1.02 at AR 1.1 up to 1.23 at AR 3, theta=1).

The encounter cross-section is pure kinematics -- no force acts before first
contact -- so it depends on the translational/rotational temperature ratio and
the shape only, and is identical across restitution coefficients.  It is stored
as the ratio r(theta, AR) = sigma_enc / sigma_c, which is smooth in log theta and
AR, and multiplied back onto the v1 polynomial at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EncounterClock:
    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text())
        if data.get("schema") != "encounter-cross-section-v1":
            raise ValueError(f"{path} is not an encounter cross-section table")
        self.aspect_ratios = np.array(sorted(float(key) for key in data["ratio"]))
        self.tables = []
        for ar in self.aspect_ratios:
            row = data["ratio"][f"{ar:.4f}"]
            theta = np.asarray(row["theta"], dtype=float)
            ratio = np.asarray(row["sigma_enc_over_sigma_c"], dtype=float)
            order = np.argsort(theta)
            self.tables.append((np.log(theta[order]), ratio[order]))
        self.path = str(path)

    def _ratio_at(self, index: int, log_theta: float) -> float:
        grid, ratio = self.tables[index]
        # Flat extrapolation in theta: the ratio saturates at both ends
        # (translation- or rotation-dominated approach).
        return float(np.interp(log_theta, grid, ratio))

    def ratio(self, theta: float, aspect_ratio: float) -> float:
        ar = float(aspect_ratio)
        if not self.aspect_ratios[0] - 1e-9 <= ar <= self.aspect_ratios[-1] + 1e-9:
            raise ValueError(f"AR={ar} outside the encounter table "
                             f"[{self.aspect_ratios[0]}, {self.aspect_ratios[-1]}]")
        log_theta = float(np.log(max(theta, 1.0e-12)))
        upper = int(np.searchsorted(self.aspect_ratios, ar).clip(1, len(self.aspect_ratios) - 1))
        lower = upper - 1
        span = self.aspect_ratios[upper] - self.aspect_ratios[lower]
        blend = 0.0 if span <= 0 else (ar - self.aspect_ratios[lower]) / span
        blend = min(max(blend, 0.0), 1.0)
        return ((1.0 - blend) * self._ratio_at(lower, log_theta)
                + blend * self._ratio_at(upper, log_theta))
