"""Strict runtime loader for routing16, VSS, and direction-only artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from coll_models_v2.direction_library import DirectionLibrary, conditioning
from coll_models_v2.surfaces import SplineSurface
from dsmc_v2_contracts import FEATURE_NAMES


class MicroscopicClosure:
    def __init__(self, routing_path: str | Path, vss_path: str | Path,
                 direction_path: str | Path):
        self.routing = json.loads(Path(routing_path).read_text())
        self.vss = json.loads(Path(vss_path).read_text())
        if self.routing.get("schema_version") != "2.1.0" \
                or self.routing.get("artifact_type") != "routing16_v2":
            raise ValueError("not a routing16_v2 artifact")
        if self.routing.get("feature_order") != list(FEATURE_NAMES):
            raise ValueError("routing feature ordering differs from runtime contract")
        if self.vss.get("inputs") != ["alpha", "aspect_ratio"] \
                or "p_eta" not in self.vss.get("forbidden_inputs", []):
            raise ValueError("VSS artifact violates the direction-only contract")
        self.surfaces = {name: SplineSurface.from_dict(payload)
                         for name, payload in self.routing.get("surfaces", {}).items()}
        self.vss_surfaces = {name: SplineSurface.from_dict(payload)
                             for name, payload in self.vss.get("surfaces", {}).items()}
        self.direction = DirectionLibrary.load(str(direction_path))

    def _routing_quantity(self, name: str, alpha: float, theta: float,
                          aspect_ratio: float) -> float:
        hull = self.routing["design_hull"]
        for value, key in ((alpha, "alpha"), (theta, "theta"),
                           (aspect_ratio, "aspect_ratio")):
            if value < hull[key][0] - 1.0e-12 or value > hull[key][1] + 1.0e-12:
                raise ValueError(f"routing query {key}={value} is outside the design hull")
        coordinate = np.array([[1.0 - alpha * alpha, np.log(theta), np.log(aspect_ratio)]])
        if name in self.surfaces:
            return float(self.surfaces[name].evaluate(coordinate)[0])
        nodes = [node for node in self.routing["nodes"] if node["alpha"] < 1.0]
        points = np.array([[1.0 - n["alpha"]**2, np.log(n["theta"]),
                            np.log(n["aspect_ratio"])] for n in nodes])
        values = np.array([n["quantities"][name]["estimate"] for n in nodes])
        exact = np.flatnonzero(np.all(np.isclose(points, coordinate[0], atol=1.0e-12), axis=1))
        if len(exact):
            return float(values[exact[0]])
        if len(points) < 4:
            raise ValueError(f"no exact {name} node and insufficient grid for interpolation")
        value = float(LinearNDInterpolator(points, values)(coordinate[0]))
        if not np.isfinite(value):
            raise ValueError("routing query lies outside the sampled data hull")
        return value

    def routing_fraction(self, alpha: float, theta: float, aspect_ratio: float,
                         features: np.ndarray) -> float:
        fc = self._routing_quantity("F_C", alpha, theta, aspect_ratio)
        beta = np.array([
            self._routing_quantity(f"beta_ctc_{name}", alpha, theta, aspect_ratio)
            for name in FEATURE_NAMES
        ])
        # f_tr is a modal production ratio, not a probability.  The unchanged
        # v1 energy update permits values outside [0,1] to represent energy
        # transfer between modes while preserving the sampled total loss.
        value = fc * (1.0 + float(np.dot(beta, features)))
        if not np.isfinite(value):
            raise ValueError("non-finite 16-moment routing response")
        return float(value)

    def alpha_eff(self, alpha: float, aspect_ratio: float) -> float:
        coordinate = np.array([[1.0 - alpha * alpha, np.log(aspect_ratio)]])
        if "alpha_eff" in self.vss_surfaces:
            return float(self.vss_surfaces["alpha_eff"].evaluate(coordinate)[0])
        rows = self.vss["rows"]
        points = np.array([[row["alpha"], row["aspect_ratio"]] for row in rows])
        values = np.array([row["alpha_eff"]["estimate"] for row in rows])
        exact = np.flatnonzero(np.all(np.isclose(points, [alpha, aspect_ratio], atol=1.0e-12), axis=1))
        if len(exact):
            return float(values[exact[0]])
        value = float(LinearNDInterpolator(points, values)(alpha, aspect_ratio))
        if not np.isfinite(value):
            raise ValueError("VSS query lies outside the sampled (alpha,AR) hull")
        return value

    def spin_directions(self, alpha: float, theta: float, aspect_ratio: float,
                        c1, c2, w1, w2, u1, u2, outgoing_fractions,
                        mass: float, inertia: float, rng: np.random.Generator) -> np.ndarray:
        query, frame = conditioning(alpha, theta, aspect_ratio, c1, c2, w1, w2,
                                    u1, u2, outgoing_fractions, mass, inertia)
        for _ in range(16):
            donor = self.direction.select(query, rng, neighbours=64).reshape(2, 3)
            lab = donor @ frame
            projected = np.array([lab[0] - np.dot(lab[0], u1) * u1,
                                  lab[1] - np.dot(lab[1], u2) * u2])
            if np.all(np.linalg.norm(projected, axis=1) > 1.0e-10):
                return projected / np.linalg.norm(projected, axis=1)[:, None]
        raise RuntimeError("rotational-direction donors were tangent-degenerate")
