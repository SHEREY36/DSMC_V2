"""Regularized tensor-product cubic regression-spline surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _basis_1d(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pieces = [np.ones_like(x), x, x * x, x**3]
    pieces.extend(np.maximum(x - knot, 0.0) ** 3 for knot in knots)
    return np.column_stack(pieces)


def _tensor_design(coordinates: np.ndarray, knots: list[np.ndarray]) -> np.ndarray:
    basis = [_basis_1d(coordinates[:, axis], knots[axis])
             for axis in range(coordinates.shape[1])]
    design = basis[0]
    for current in basis[1:]:
        design = np.einsum("ni,nj->nij", design, current).reshape(len(coordinates), -1)
    return design


@dataclass
class SplineSurface:
    variables: list[str]
    lower: list[float]
    upper: list[float]
    knots: list[list[float]]
    coefficients: list[float]
    ridge: float
    schema_version: str = "2.0.0"

    def evaluate(self, coordinates: np.ndarray) -> np.ndarray:
        coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))
        lower, upper = np.asarray(self.lower), np.asarray(self.upper)
        if np.any(coordinates < lower - 1.0e-12) or np.any(coordinates > upper + 1.0e-12):
            raise ValueError("surface query lies outside the fitted design hull")
        scaled = (coordinates - lower) / np.maximum(upper - lower, 1.0e-30)
        design = _tensor_design(scaled, [np.asarray(k) for k in self.knots])
        return design @ np.asarray(self.coefficients)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "SplineSurface":
        return cls(**payload)


def fit_surface(coordinates: np.ndarray, values: np.ndarray, variables: list[str],
                standard_errors: np.ndarray | None = None,
                ridge: float = 1.0e-4) -> SplineSurface:
    coordinates = np.asarray(coordinates, dtype=float)
    values = np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or len(coordinates) != len(values):
        raise ValueError("coordinates must be (N,D) and match values")
    if coordinates.shape[1] != len(variables):
        raise ValueError("variable count does not match coordinate dimension")
    finite = np.isfinite(values) & np.all(np.isfinite(coordinates), axis=1)
    coordinates, values = coordinates[finite], values[finite]
    if len(values) < 2:
        raise ValueError("at least two finite surface nodes are required")
    lower, upper = np.min(coordinates, axis=0), np.max(coordinates, axis=0)
    if np.any(upper <= lower):
        raise ValueError("every fitted coordinate must span at least two values")
    scaled = (coordinates - lower) / (upper - lower)
    knots = []
    for axis in range(scaled.shape[1]):
        unique = np.unique(scaled[:, axis])
        knots.append(np.quantile(unique, (1.0 / 3.0, 2.0 / 3.0)) if len(unique) >= 5 else np.array([]))
    design = _tensor_design(scaled, knots)
    if standard_errors is None:
        weight = np.ones(len(values))
    else:
        error = np.asarray(standard_errors, dtype=float)[finite]
        positive = error[np.isfinite(error) & (error > 0.0)]
        floor = np.median(positive) * 0.1 if len(positive) else 1.0
        weight = 1.0 / np.maximum(np.where(np.isfinite(error), error, floor), floor)
    lhs = (design * weight[:, None]).T @ (design * weight[:, None])
    rhs = (design * weight[:, None]).T @ (values * weight)
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(lhs + penalty, rhs)
    return SplineSurface(variables, lower.tolist(), upper.tolist(),
                         [k.tolist() for k in knots], coefficients.tolist(), float(ridge))


def transformed_coordinates(alpha: np.ndarray, theta: np.ndarray,
                            aspect_ratio: np.ndarray) -> np.ndarray:
    return np.column_stack((1.0 - np.asarray(alpha) ** 2,
                            np.log(np.asarray(theta)),
                            np.log(np.asarray(aspect_ratio))))

