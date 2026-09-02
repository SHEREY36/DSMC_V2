"""Canonical invariant basis shared by CTC estimation and DSMC runtime.

The production closure uses fourteen O(3)-invariant cell features. Two
additional invariants are reported as diagnostics but are deliberately kept
outside the deployed natural-parameter correction.
"""

from __future__ import annotations

import math
import numpy as np


FEATURE_NAMES = (
    "a2_tr", "a2_rot", "a11", "A_cu",
    "PiPi", "QQ", "RtRt", "PiQ", "PiRt", "QRt",
    "qtr2", "qrot2", "qtr_qrot", "W2",
)
DIAGNOSTIC_NAMES = ("Acw2", "vx2")
ALL_INVARIANT_NAMES = FEATURE_NAMES + DIAGNOSTIC_NAMES
LEGACY_FEATURE_NAMES = (
    "a2_tr", "a2_rot", "a11", "PiPi", "RR", "QQ", "PiR", "PiQ", "RQ",
    "qtrqtr", "qrotqrot", "qtrqrot", "a3_tr", "a3_rot", "a21", "a12",
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
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (
        1.5 - x,
        15.0 / 8.0 - 2.5 * x + 0.5 * x * x,
        35.0 / 16.0 - 35.0 * x / 8.0 + 7.0 * x * x / 4.0 - x**3 / 6.0,
        1.0 - y,
        1.0 - 2.0 * y + 0.5 * y * y,
        1.0 - 3.0 * y + 1.5 * y * y - y**3 / 6.0,
    )


def _unit_rows(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values, axis=1)
    if np.any(norm <= 1.0e-30):
        raise ValueError(f"{name} contains a zero vector")
    return values / norm[:, None]


def _particle_moments(c: np.ndarray, w: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
    """Per-particle contributions whose expectations define the invariants."""
    x = np.einsum("ni,ni->n", c, c)
    y = np.einsum("ni,ni->n", w, w)
    ident = np.eye(3)
    pi = 2.0 * (np.einsum("ni,nj->nij", c, c) - x[:, None, None] * ident / 3.0)
    rr = 3.0 * (np.einsum("ni,nj->nij", w, w) - y[:, None, None] * ident / 3.0)
    qq = 0.5 * (3.0 * np.einsum("ni,nj->nij", u, u) - ident)
    rt = rr + qq
    return {
        "x": x,
        "y": y,
        "pi": pi,
        "qq": qq,
        "rt": rt,
        "qtr": 0.8 * c * (x[:, None] - 2.5),
        "qrot": 2.0 * c * (y[:, None] - 1.0),
        "w": w,
        "acw": np.einsum("ni,ni->n", c, w),
        "vx": np.cross(c, w),
        "acu": np.einsum("ni,ni->n", c, u) ** 2 - x / 3.0,
    }


def _pair_contract(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nij->n", a1, a2)


def _pair_dot(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ni->n", a1, a2)


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
    """Return pair U-kernels for the fourteen production invariants.

    Averaging these kernels over independent proposal pairs gives the same
    population invariants as :func:`cell_features`. They are used for
    proposal-balance diagnostics; coefficient identification uses directly
    generated excitation ensembles rather than score reweighting.
    """
    arrays = [np.atleast_2d(np.asarray(value, dtype=float)) for value in
              (c1, c2, omega1, omega2, u1, u2)]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("all pair arrays must have the same row count")
    if velocity_scale <= 0.0 or omega_scale <= 0.0:
        raise ValueError("velocity_scale and omega_scale must be positive")
    c1, c2 = arrays[0] / velocity_scale, arrays[1] / velocity_scale
    w1, w2 = arrays[2] / omega_scale, arrays[3] / omega_scale
    u1, u2 = _unit_rows(arrays[4], "u1"), _unit_rows(arrays[5], "u2")
    m1, m2 = _particle_moments(c1, w1, u1), _particle_moments(c2, w2, u2)
    out = np.empty((len(c1), len(FEATURE_NAMES)), dtype=float)
    out[:, 0] = 0.5 * ((4.0 / 15.0) * m1["x"] ** 2 - 1.0
                       + (4.0 / 15.0) * m2["x"] ** 2 - 1.0)
    out[:, 1] = 0.5 * (0.5 * m1["y"] ** 2 - 1.0
                       + 0.5 * m2["y"] ** 2 - 1.0)
    out[:, 2] = 0.5 * ((2.0 / 3.0) * m1["x"] * m1["y"] - 1.0
                       + (2.0 / 3.0) * m2["x"] * m2["y"] - 1.0)
    out[:, 3] = 0.5 * (m1["acu"] + m2["acu"])
    out[:, 4] = _pair_contract(m1["pi"], m2["pi"]) / 8.0
    out[:, 5] = _pair_contract(m1["qq"], m2["qq"]) / 8.0
    out[:, 6] = _pair_contract(m1["rt"], m2["rt"]) / 8.0
    out[:, 7] = (_pair_contract(m1["pi"], m2["qq"])
                  + _pair_contract(m2["pi"], m1["qq"])) / 8.0
    out[:, 8] = (_pair_contract(m1["pi"], m2["rt"])
                  + _pair_contract(m2["pi"], m1["rt"])) / 8.0
    out[:, 9] = (_pair_contract(m1["qq"], m2["rt"])
                  + _pair_contract(m2["qq"], m1["rt"])) / 8.0
    out[:, 10] = _pair_dot(m1["qtr"], m2["qtr"])
    out[:, 11] = _pair_dot(m1["qrot"], m2["qrot"])
    out[:, 12] = 0.5 * (_pair_dot(m1["qtr"], m2["qrot"])
                        + _pair_dot(m2["qtr"], m1["qrot"]))
    out[:, 13] = _pair_dot(m1["w"], m2["w"])
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


def _normalised_state(velocity: np.ndarray, omega: np.ndarray, axis: np.ndarray,
                      mass: float, moi_perpendicular: float) -> tuple[np.ndarray, ...]:
    velocity = np.asarray(velocity, dtype=float)
    omega = np.asarray(omega, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if velocity.shape != omega.shape or velocity.shape != axis.shape \
            or velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("velocity, omega and axis must all have shape (N,3)")
    if len(velocity) < 2:
        raise ValueError("at least two particles are required")
    axis = _unit_rows(axis, "axis")
    cpec = velocity - np.mean(velocity, axis=0)
    ttr = mass * np.einsum("ni,ni->", cpec, cpec) / (3.0 * len(cpec))
    trot = moi_perpendicular * np.einsum("ni,ni->", omega, omega) / (2.0 * len(omega))
    if ttr <= 0.0 or trot <= 0.0:
        raise ValueError("cell translational and rotational temperatures must be positive")
    c = cpec / np.sqrt(2.0 * ttr / mass)
    w = omega / np.sqrt(2.0 * trot / moi_perpendicular)
    return c, w, axis


def cell_invariants(
    velocity: np.ndarray,
    omega: np.ndarray,
    axis: np.ndarray,
    mass: float = 1.0,
    moi_perpendicular: float = 1.0,
    sphere: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fourteen production features and two diagnostics."""
    c, w, u = _normalised_state(velocity, omega, axis, mass, moi_perpendicular)
    moments = _particle_moments(c, w, u)
    if sphere:
        moments["rt"] -= moments["qq"]
        moments["qq"][:] = 0.0
        moments["acu"][:] = 0.0
    x, y = moments["x"], moments["y"]
    features = np.array([
        (4.0 / 15.0) * np.mean(x * x) - 1.0,
        0.5 * np.mean(y * y) - 1.0,
        (2.0 / 3.0) * np.mean(x * y) - 1.0,
        np.mean(moments["acu"]),
        _u_contraction(moments["pi"], moments["pi"]) / 8.0,
        _u_contraction(moments["qq"], moments["qq"]) / 8.0,
        _u_contraction(moments["rt"], moments["rt"]) / 8.0,
        _u_contraction(moments["pi"], moments["qq"]) / 4.0,
        _u_contraction(moments["pi"], moments["rt"]) / 4.0,
        _u_contraction(moments["qq"], moments["rt"]) / 4.0,
        _u_dot(moments["qtr"], moments["qtr"]),
        _u_dot(moments["qrot"], moments["qrot"]),
        _u_dot(moments["qtr"], moments["qrot"]),
        _u_dot(moments["w"], moments["w"]),
    ], dtype=float)
    acw = moments["acw"][:, None]
    diagnostics = np.array([
        _u_dot(acw, acw),
        _u_dot(moments["vx"], moments["vx"]),
    ], dtype=float)
    return features, diagnostics


def cell_features(*args, **kwargs) -> np.ndarray:
    """Compatibility wrapper returning only the deployed fourteen features."""
    return cell_invariants(*args, **kwargs)[0]


def legacy_cell_features(velocity: np.ndarray, omega: np.ndarray, axis: np.ndarray,
                         mass: float = 1.0, moi_perpendicular: float = 1.0,
                         sphere: bool = False) -> np.ndarray:
    """The frozen schema-2.1 feature vector for complete legacy A/B runs."""
    c, w, u = _normalised_state(velocity, omega, axis, mass, moi_perpendicular)
    x = np.einsum("ni,ni->n", c, c)
    y = np.einsum("ni,ni->n", w, w)
    s1, s2, s3, l1, l2, l3 = radial_polynomials(x, y)
    ident = np.eye(3)
    pi = 2.0 * (np.einsum("ni,nj->nij", c, c) - x[:, None, None] * ident / 3.0)
    rr = 3.0 * (np.einsum("ni,nj->nij", w, w) - y[:, None, None] * ident / 3.0)
    qq = 0.5 * (3.0 * np.einsum("ni,nj->nij", u, u) - ident)
    if sphere:
        qq[:] = 0.0
    qt, qr = 0.8 * c * (x[:, None] - 2.5), 2.0 * c * (y[:, None] - 1.0)
    return np.array([
        (8.0 / 15.0) * np.mean(s2), np.mean(l2), (2.0 / 3.0) * np.mean(s1 * l1),
        _u_contraction(pi, pi) / 8.0, _u_contraction(rr, rr) / 8.0,
        _u_contraction(qq, qq) / 8.0, _u_contraction(pi, rr) / 4.0,
        _u_contraction(pi, qq) / 4.0, _u_contraction(rr, qq) / 4.0,
        _u_dot(qt, qt), _u_dot(qr, qr), _u_dot(qt, qr),
        (16.0 / 35.0) * np.mean(-s3), np.mean(l3),
        (8.0 / 15.0) * np.mean(s2 * l1), (2.0 / 3.0) * np.mean(s1 * l2),
    ], dtype=float)
