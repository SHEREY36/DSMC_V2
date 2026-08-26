"""Strict loader for collision_operator_v2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from coll_models_v2.energy_library import EnergyLibrary
from coll_models_v2.pair_clock import PairClockModel, invariants_from_state
from coll_models_v2.surfaces import SplineSurface


def _logistic(value: float) -> float:
    value = float(np.clip(value, -35.0, 35.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def _logit(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"logit input must be in (0,1), got {value}")
    return float(np.log(value / (1.0 - value)))


class CollisionArtifactV2:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("schema_version") != "2.0.0" \
                or self.manifest.get("artifact_type") != "collision_operator_v2":
            raise ValueError("not a collision_operator_v2 artifact")
        clock = json.loads((self.directory / "pair_clock_v2.json").read_text())
        self.clock_models = {float(ar): PairClockModel.from_dict(payload)
                             for ar, payload in clock.items()}
        self.energy = EnergyLibrary.load(str(self.directory / "energy_library_v2.npz"))
        vss = json.loads((self.directory / "vss_rank2_v2.json").read_text())
        if vss.get("inputs") != ["alpha", "aspect_ratio"] or "theta" not in vss.get("forbidden_inputs", []):
            raise ValueError("VSS artifact violates the direction-only input contract")
        self.vss_rows = vss["rows"]
        reduced = json.loads((self.directory / "reduced_surfaces_v2.json").read_text())
        self.surfaces = {name: SplineSurface.from_dict(payload)
                         for name, payload in reduced.get("surfaces", {}).items()}

    def _clock(self, aspect_ratio: float) -> PairClockModel:
        keys = np.array(list(self.clock_models))
        match = np.flatnonzero(np.isclose(keys, aspect_ratio, rtol=0.0, atol=1.0e-12))
        if not len(match):
            raise ValueError(f"pair clock unavailable for AR={aspect_ratio}; no AR extrapolation is allowed")
        return self.clock_models[float(keys[match[0]])]

    def proposal_area(self, aspect_ratio: float) -> float:
        return self._clock(aspect_ratio).proposal_area

    def pair_cross_section(self, c1, c2, w1, w2, u1, u2,
                           aspect_ratio: float) -> float:
        invariants = invariants_from_state(c1, c2, w1, w2, u1, u2)
        return float(self._clock(aspect_ratio).cross_section_from_invariants(invariants)[0])

    def alpha_eff(self, alpha: float, aspect_ratio: float) -> float:
        points = np.array([[row["alpha"], row["aspect_ratio"]] for row in self.vss_rows])
        values = np.array([row["alpha_eff"]["estimate"] for row in self.vss_rows], dtype=float)
        exact = np.flatnonzero(np.all(np.isclose(points, [alpha, aspect_ratio], atol=1.0e-12), axis=1))
        if len(exact):
            return float(values[exact[0]])
        same_ar = np.flatnonzero(np.isclose(points[:, 1], aspect_ratio, atol=1.0e-12))
        if len(same_ar) >= 2:
            order = same_ar[np.argsort(points[same_ar, 0])]
            if alpha < points[order[0], 0] or alpha > points[order[-1], 0]:
                raise ValueError("VSS alpha query lies outside the design hull")
            return float(np.interp(alpha, points[order, 0], values[order]))
        if len(points) < 3:
            raise ValueError("insufficient VSS nodes for two-dimensional interpolation")
        value = float(LinearNDInterpolator(points, values)(alpha, aspect_ratio))
        if not np.isfinite(value):
            raise ValueError("VSS query lies outside the design hull")
        return value

    def energy_conditioning(self, c1, c2, w1, w2, u1, u2,
                            alpha: float, theta: float, aspect_ratio: float) -> np.ndarray:
        g = np.asarray(c2) - np.asarray(c1)
        gmag = np.linalg.norm(g)
        gh = g / max(gmag, 1.0e-30)
        return np.array([
            alpha, np.log(theta), np.log(aspect_ratio), np.log(max(gmag, 1.0e-12)),
            np.log1p(np.dot(w1, w1) + np.dot(w2, w2)), abs(np.dot(gh, u1)),
            abs(np.dot(gh, u2)), np.dot(u1, u2) ** 2,
        ])

    def sample_pair_outcome(self, c1, c2, w1, w2, u1, u2, alpha, theta,
                            aspect_ratio, rng) -> np.ndarray:
        query = self.energy_conditioning(c1, c2, w1, w2, u1, u2,
                                         alpha, theta, aspect_ratio)
        return self.energy.sample(query, rng)

    def _surface(self, name: str, alpha: float, theta: float,
                 aspect_ratio: float) -> float:
        if name not in self.surfaces:
            raise ValueError(f"reduced surface {name!r} is unavailable")
        coordinate = np.array([[1.0 - alpha * alpha, np.log(theta), np.log(aspect_ratio)]])
        return float(self.surfaces[name].evaluate(coordinate)[0])

    def reduced_means(self, alpha: float, theta: float, aspect_ratio: float,
                      features: np.ndarray) -> tuple[float, float, float]:
        area = self.proposal_area(aspect_ratio)
        sigma0 = self._surface("sigma0", alpha, theta, aspect_ratio)
        eta_sigma = np.array([self._surface(f"eta_sigma_{name}", alpha, theta, aspect_ratio)
                              for name in _feature_names()])
        sigma = area * _logistic(_logit(sigma0 / area) + np.dot(eta_sigma, features))
        if alpha >= 1.0:
            return sigma, 0.0, 1.0 if np.isclose(aspect_ratio, 1.0) else np.nan
        gamma0 = self._surface("Gamma0", alpha, theta, aspect_ratio)
        ftr0 = self._surface("Ftr0", alpha, theta, aspect_ratio)
        eta_gamma = np.array([self._surface(f"eta_gamma_{name}", alpha, theta, aspect_ratio)
                              for name in _feature_names()])
        gamma = _logistic(_logit(gamma0) + np.dot(eta_gamma, features))
        if np.isclose(aspect_ratio, 1.0):
            ftr = 1.0
        else:
            eta_ftr = np.array([self._surface(f"eta_ftr_{name}", alpha, theta, aspect_ratio)
                                for name in _feature_names()])
            ftr = _logistic(_logit(ftr0) + np.dot(eta_ftr, features))
        return sigma, gamma, ftr


def _feature_names():
    from dsmc_v2_contracts import FEATURE_NAMES
    return FEATURE_NAMES

