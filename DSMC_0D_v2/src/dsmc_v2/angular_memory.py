"""Angular scattering law with incoming-partition memory.

For one CTC encounter let c = ghat.ghat', z the incoming and z' the outgoing
translational share.  The law is the conditional I-projection

    p(c | z, z') = exp[a c + b P2(c)] / Z(a, b),
    a = eta1 + xi z' + rho1 z,     b = eta2 + zeta z' + rho2 z,

fitted on the post-collision-energy-tilted encounter measure (weight E_f).
Because z' P2(c) is a sufficient statistic, the fitted law reproduces
E[E_f z' P2(c)] -- the collisional production of the traceless stress -- and the
z terms carry that production to non-equilibrium incoming states.

Parameters are stored per artifact node.  Between nodes the runtime samples
the mixture of the neighbouring node laws with the same physical stencil and
weights as the energy kernel (interpolation of distributions, not of natural
parameters).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PARAMETER_NAMES = ("eta1", "xi", "rho1", "eta2", "zeta", "rho2")


class AngularMemoryTable:
    """Per-node parameters, sampled on the closure's own physical stencil."""

    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text())
        if data.get("schema") != "angular-memory-v1":
            raise ValueError(f"{path} is not an angular-memory table")
        if tuple(data["parameter_names"]) != PARAMETER_NAMES:
            raise ValueError("angular-memory parameter order changed")
        self.by_node = {}
        for node in data["nodes"]:
            key = self._key(node["alpha"], node["theta"], node["AR"])
            values = np.array([node[name] for name in PARAMETER_NAMES]
                              + list(node["a_support"]) + list(node["b_support"]),
                              dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite angular-memory parameters at {key}")
            self.by_node[key] = values
        self.path = str(path)
        self._node_parameters = None

    @staticmethod
    def _key(alpha, theta, aspect_ratio):
        return (round(float(alpha), 4), round(float(theta), 5), round(float(aspect_ratio), 4))

    def bind(self, coordinates: np.ndarray) -> None:
        """Align with the artifact node order; fail closed on any gap."""
        rows = []
        for alpha, theta, aspect_ratio in np.asarray(coordinates, dtype=float):
            key = self._key(alpha, theta, aspect_ratio)
            if key not in self.by_node:
                raise ValueError(f"angular-memory table has no node {key}")
            rows.append(self.by_node[key])
        self._node_parameters = np.asarray(rows)

    def stencil(self, state: dict):
        """(weight, parameters) pairs on the energy kernel's vertex stencil."""
        if self._node_parameters is None:
            raise RuntimeError("angular-memory table is not bound to the artifact")
        indices = np.asarray(state["energy_vertex_indices"], dtype=int)
        weights = np.asarray(state["energy_vertex_weights"], dtype=float)
        keep = weights > 1.0e-12
        total = float(np.sum(weights[keep]))
        return [(float(w) / total, self._node_parameters[i])
                for i, w in zip(indices[keep], weights[keep])]


def sample_cosine(stencil, z_in: float, z_out: float, rng: np.random.Generator) -> float:
    """Draw c from the stencil mixture of exp(a c + b P2(c)) laws."""
    draw = rng.random()
    parameters = stencil[-1][1]
    for weight, candidate in stencil:
        draw -= weight
        if draw <= 0.0:
            parameters = candidate
            break
    eta1, xi, rho1, eta2, zeta, rho2, a_lo, a_hi, b_lo, b_hi = parameters
    # Support gate: a node law is only evaluated inside the natural-parameter
    # range its own encounters produced; a mixture partner from a distant
    # theta node must not extrapolate the z-memory terms.
    linear = min(max(eta1 + xi * z_out + rho1 * z_in, a_lo), a_hi)
    quadratic = min(max(eta2 + zeta * z_out + rho2 * z_in, b_lo), b_hi)
    candidates = [-1.0, 1.0]
    if quadratic < 0.0:
        vertex = -linear / (3.0 * quadratic)
        if -1.0 < vertex < 1.0:
            candidates.append(vertex)
    maximum = max(linear * c + quadratic * 0.5 * (3.0 * c * c - 1.0) for c in candidates)
    for _ in range(100000):
        cosine = 2.0 * rng.random() - 1.0
        exponent = linear * cosine + quadratic * 0.5 * (3.0 * cosine * cosine - 1.0)
        if rng.random() <= np.exp(exponent - maximum):
            return cosine
    raise RuntimeError("angular-memory rejection sampler failed")


def direction_from_cosine(ghat: np.ndarray, cosine: float, rng: np.random.Generator) -> np.ndarray:
    ghat = np.asarray(ghat, dtype=float)
    ghat = ghat / max(np.linalg.norm(ghat), 1.0e-30)
    trial = np.array([1.0, 0.0, 0.0]) if abs(ghat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, ghat) * ghat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ghat, e1)
    phi = 2.0 * np.pi * rng.random()
    result = cosine * ghat + np.sqrt(max(0.0, 1.0 - cosine * cosine)) * (
        np.cos(phi) * e1 + np.sin(phi) * e2)
    return result / np.linalg.norm(result)
