"""Canonical 16-feature basis used by CTC estimation and DSMC runtime.

The score kernels are dual perturbations at the isotropic Maxwellian. Cell
features are the corresponding physical moments. Keeping both definitions in
one package prevents estimator/runtime normalization drift.
"""

from __future__ import annotations

import math

import numpy as np


FEATURE_NAMES = (
    "a2_tr", "a2_rot", "a11",
    "PiPi", "RR", "QQ", "PiR", "PiQ", "RQ",
    "qtrqtr", "qrotqrot", "qtrqrot",
    "a3_tr", "a3_rot", "a21", "a12",
)


def _tensor_basis() -> np.ndarray:
    basis = np.zeros((5, 3, 3), dtype=float)
    basis[0] = np.diag([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    basis[1] = np.diag([1.0, 1.0, -2.0]) / math.sqrt(6.0)
    basis[2, 0, 1] = basis[2, 1, 0] = 1.0 / math.sqrt(2.0)
    basis[3, 0, 2] = basis[3, 2, 0] = 1.0 / math.sqrt(2.0)
    basis[4, 1, 2] = basis[4, 2, 1] = 1.0 / math.sqrt(2.0)
    return basis


TENSOR_BASIS = _tensor_basis()


def radial_polynomials(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return S1,S2,S3,L1,L2,L3 in the agreed normalization."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s1 = 1.5 - x
    s2 = 15.0 / 8.0 - 2.5 * x + 0.5 * x * x
    s3 = 35.0 / 16.0 - 35.0 * x / 8.0 + 7.0 * x * x / 4.0 - x**3 / 6.0
    l1 = 1.0 - y
    l2 = 1.0 - 2.0 * y + 0.5 * y * y
    l3 = 1.0 - 3.0 * y + 1.5 * y * y - y**3 / 6.0
    return s1, s2, s3, l1, l2, l3


def _quadratic_projection(vectors: np.ndarray) -> np.ndarray:
    return np.einsum("ni,kij,nj->nk", vectors, TENSOR_BASIS, vectors, optimize=True)


def pair_score_kernel(
    c1: np.ndarray,
    c2: np.ndarray,
    omega1: np.ndarray,
    omega2: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    velocity_scale: float,
    omega_scale: float,
) -> np.ndarray:
    """Evaluate the 16 dual pair scores for one or many incoming pairs."""
    arrays = [np.atleast_2d(np.asarray(value, dtype=float)) for value in
              (c1, c2, omega1, omega2, u1, u2)]
    c1, c2, omega1, omega2, u1, u2 = arrays
    if velocity_scale <= 0.0 or omega_scale <= 0.0:
        raise ValueError("velocity_scale and omega_scale must be positive")
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("all pair arrays must have the same row count")
    c1 = c1 / velocity_scale
    c2 = c2 / velocity_scale
    omega1 = omega1 / omega_scale
    omega2 = omega2 / omega_scale
    x1 = np.einsum("ni,ni->n", c1, c1)
    x2 = np.einsum("ni,ni->n", c2, c2)
    y1 = np.einsum("ni,ni->n", omega1, omega1)
    y2 = np.einsum("ni,ni->n", omega2, omega2)
    s11, s21, s31, l11, l21, l31 = radial_polynomials(x1, y1)
    s12, s22, s32, l12, l22, l32 = radial_polynomials(x2, y2)

    pi1, pi2 = _quadratic_projection(c1), _quadratic_projection(c2)
    ww1, ww2 = _quadratic_projection(omega1), _quadratic_projection(omega2)
    uu1, uu2 = _quadratic_projection(u1), _quadratic_projection(u2)
    # These are the independently normalized spin/alignment dual scores.
    r1, r2 = (10.0 * ww1 + 5.0 * uu1) / 7.0, (10.0 * ww2 + 5.0 * uu2) / 7.0
    q1, q2 = (10.0 * ww1 + 40.0 * uu1) / 7.0, (10.0 * ww2 + 40.0 * uu2) / 7.0

    qtr1, qtr2 = c1 * (x1[:, None] - 2.5), c2 * (x2[:, None] - 2.5)
    qrot1, qrot2 = c1 * (y1[:, None] - 1.0), c2 * (y2[:, None] - 1.0)
    out = np.empty((len(c1), 16), dtype=float)
    out[:, :3] = np.column_stack((
        s21 + s22,
        l21 + l22,
        s11 * l11 + s12 * l12,
    ))
    out[:, 3] = np.mean(8.0 * pi1 * pi2, axis=1)
    out[:, 4] = np.mean(8.0 * r1 * r2, axis=1)
    out[:, 5] = np.mean(8.0 * q1 * q2, axis=1)
    out[:, 6] = np.mean(4.0 * (pi1 * r2 + r1 * pi2), axis=1)
    out[:, 7] = np.mean(4.0 * (pi1 * q2 + q1 * pi2), axis=1)
    out[:, 8] = np.mean(4.0 * (r1 * q2 + q1 * r2), axis=1)
    out[:, 9] = np.mean(qtr1 * qtr2, axis=1)
    out[:, 10] = np.mean(qrot1 * qrot2, axis=1)
    out[:, 11] = np.mean(qtr1 * qrot2 + qrot1 * qtr2, axis=1)
    out[:, 12:] = np.column_stack((
        -s31 - s32,
        l31 + l32,
        s21 * l11 + s22 * l12,
        s11 * l21 + s12 * l22,
    ))
    return out


def _u_contraction(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n < 2:
        raise ValueError("at least two particles are required for U-statistics")
    return float((np.sum(a, axis=0).ravel() @ np.sum(b, axis=0).ravel()
                  - np.einsum("nij,nij->", a, b)) / (n * (n - 1)))


def _u_dot(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    return float((np.sum(a, axis=0) @ np.sum(b, axis=0)
                  - np.einsum("ni,ni->", a, b)) / (n * (n - 1)))


def cell_features(
    velocity: np.ndarray,
    omega: np.ndarray,
    axis: np.ndarray,
    mass: float = 1.0,
    moi_perpendicular: float = 1.0,
    sphere: bool = False,
) -> np.ndarray:
    """Return unbiased cell estimates of the 16 closure variables."""
    velocity = np.asarray(velocity, dtype=float)
    omega = np.asarray(omega, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if velocity.shape != omega.shape or velocity.shape != axis.shape or velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("velocity, omega and axis must all have shape (N,3)")
    n = len(velocity)
    if n < 2:
        raise ValueError("at least two particles are required")
    cpec = velocity - np.mean(velocity, axis=0)
    ttr = mass * np.einsum("ni,ni->", cpec, cpec) / (3.0 * n)
    trot = moi_perpendicular * np.einsum("ni,ni->", omega, omega) / (2.0 * n)
    if ttr <= 0.0 or trot <= 0.0:
        raise ValueError("cell translational and rotational temperatures must be positive")
    c = cpec / math.sqrt(2.0 * ttr / mass)
    w = omega / math.sqrt(2.0 * trot / moi_perpendicular)
    x = np.einsum("ni,ni->n", c, c)
    y = np.einsum("ni,ni->n", w, w)
    s1, s2, s3, l1, l2, l3 = radial_polynomials(x, y)
    ident = np.eye(3)
    pi = 2.0 * (np.einsum("ni,nj->nij", c, c) - x[:, None, None] * ident / 3.0)
    rr = 3.0 * (np.einsum("ni,nj->nij", w, w) - y[:, None, None] * ident / 3.0)
    qq = 0.5 * (3.0 * np.einsum("ni,nj->nij", axis, axis) - ident)
    if sphere:
        qq[:] = 0.0
    qt = 0.8 * c * (x[:, None] - 2.5)
    qr = 2.0 * c * (y[:, None] - 1.0)
    return np.array([
        (8.0 / 15.0) * np.mean(s2),
        np.mean(l2),
        (2.0 / 3.0) * np.mean(s1 * l1),
        _u_contraction(pi, pi) / 8.0,
        _u_contraction(rr, rr) / 8.0,
        _u_contraction(qq, qq) / 8.0,
        _u_contraction(pi, rr) / 4.0,
        _u_contraction(pi, qq) / 4.0,
        _u_contraction(rr, qq) / 4.0,
        _u_dot(qt, qt),
        _u_dot(qr, qr),
        _u_dot(qt, qr),
        (16.0 / 35.0) * np.mean(-s3),
        np.mean(l3),
        (8.0 / 15.0) * np.mean(s2 * l1),
        (2.0 / 3.0) * np.mean(s1 * l2),
    ], dtype=float)

