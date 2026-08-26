"""Rotationally invariant additive spline-logistic pair cross-section model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from dsmc_v2_contracts.io import RunDataV2


def invariant_pair_inputs(run: RunDataV2) -> np.ndarray:
    values = np.asarray(run.attempts["values"])
    c1, c2 = values[:, 0:3], values[:, 3:6]
    w1, w2 = values[:, 6:9], values[:, 9:12]
    u1, u2 = values[:, 12:15], values[:, 15:18]
    return invariants_from_state(c1, c2, w1, w2, u1, u2)


def invariants_from_state(c1, c2, w1, w2, u1, u2) -> np.ndarray:
    c1, c2, w1, w2, u1, u2 = [np.atleast_2d(np.asarray(v, dtype=float))
                               for v in (c1, c2, w1, w2, u1, u2)]
    g = c2 - c1
    gmag = np.linalg.norm(g, axis=1)
    gh = g / np.maximum(gmag[:, None], 1.0e-30)
    w1mag, w2mag = np.linalg.norm(w1, axis=1), np.linalg.norm(w2, axis=1)
    return np.column_stack((
        np.log(np.maximum(gmag, 1.0e-12)),
        np.abs(np.einsum("ni,ni->n", gh, u1)),
        np.abs(np.einsum("ni,ni->n", gh, u2)),
        np.einsum("ni,ni->n", u1, u2) ** 2,
        np.log1p(w1mag), np.log1p(w2mag),
        np.einsum("ni,ni->n", w1, w2) /
        np.maximum(w1mag * w2mag, 1.0e-12),
    ))


def _spline_design(z: np.ndarray, knots: np.ndarray) -> np.ndarray:
    pieces = [np.ones((len(z), 1)), z, z * z]
    for feature in range(z.shape[1]):
        for knot in knots[feature]:
            pieces.append(np.maximum(z[:, feature:feature + 1] - knot, 0.0) ** 3)
    return np.column_stack(pieces)


@dataclass
class PairClockModel:
    feature_mean: list[float]
    feature_scale: list[float]
    knots: list[list[float]]
    coefficients: list[float]
    proposal_area: float
    ridge: float
    schema_version: str = "2.0.0"

    def probability_from_invariants(self, values: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(values, dtype=float))
        z = (values - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        design = _spline_design(z, np.asarray(self.knots))
        eta = np.clip(design @ np.asarray(self.coefficients), -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-eta))

    def cross_section_from_invariants(self, values: np.ndarray) -> np.ndarray:
        return self.proposal_area * self.probability_from_invariants(values)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "PairClockModel":
        return cls(**payload)


def fit_pair_clock(runs: list[RunDataV2], ridge: float = 1.0e-3,
                   max_iter: int = 80) -> PairClockModel:
    if not runs:
        raise ValueError("at least one CTC run is required")
    x = np.concatenate([invariant_pair_inputs(run) for run in runs])
    y = np.concatenate([np.asarray(run.attempts["hit"], dtype=float) for run in runs])
    areas = np.array([float(run.metadata["proposal_area"]) for run in runs])
    if not np.allclose(areas, areas[0], rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("pair-clock runs must share one proposal area/AR")
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    scale[scale < 1.0e-10] = 1.0
    z = (x - mean) / scale
    knots = np.quantile(z, (0.25, 0.5, 0.75), axis=0).T
    design = _spline_design(z, knots)
    beta = np.zeros(design.shape[1])
    penalty = np.eye(len(beta)) * float(ridge)
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(probability * (1.0 - probability), 1.0e-8)
        hessian = design.T @ (weight[:, None] * design) + penalty
        gradient = design.T @ (y - probability) - penalty @ beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.linalg.norm(step) <= 1.0e-9 * (1.0 + np.linalg.norm(beta)):
            break
    return PairClockModel(mean.tolist(), scale.tolist(), knots.tolist(), beta.tolist(),
                          float(areas[0]), float(ridge))

