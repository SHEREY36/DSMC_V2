"""Constrained full post-state reconstruction.

VSS determines only the new relative-velocity direction. The energy library
determines retained energy and its translational/rotational composition. This
solver composes the two while enforcing linear/angular momentum, tangent spin,
and nonnegative energy exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coll_models_v2.energy_library import pair_frame


@dataclass
class ReconstructionResult:
    velocity1: np.ndarray
    velocity2: np.ndarray
    omega1: np.ndarray
    omega2: np.ndarray
    energy_error: float
    angular_momentum_error: float


def _tangent_basis(axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, axis)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    first = trial - np.dot(trial, axis) * axis
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(axis, first)))


def _solve_spins(target_spin: np.ndarray, rotational_energy: float,
                 preferred_first_fraction: float, u1: np.ndarray, u2: np.ndarray,
                 moi: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    b1, b2 = _tangent_basis(u1), _tangent_basis(u2)
    matrix = np.column_stack((b1, b2))
    target = target_spin / moi
    x0, _, _, _ = np.linalg.lstsq(matrix, target, rcond=1.0e-12)
    if np.linalg.norm(matrix @ x0 - target) > 1.0e-9 * max(1.0, np.linalg.norm(target)):
        return None
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular > 1.0e-11 * singular[0])) if len(singular) else 0
    null = vh[rank:].T
    target_norm2 = rotational_energy / moi
    remaining = target_norm2 - np.dot(x0, x0)
    if remaining < -1.0e-10 * max(1.0, target_norm2) or null.shape[1] == 0:
        if abs(remaining) <= 1.0e-10 * max(1.0, target_norm2):
            candidate = x0
            return b1 @ candidate[:2], b2 @ candidate[2:]
        return None
    radius = np.sqrt(max(0.0, remaining))
    candidates = []
    if null.shape[1] == 1:
        candidates = [x0 + radius * null[:, 0], x0 - radius * null[:, 0]]
    else:
        for _ in range(32):
            direction = rng.normal(size=null.shape[1])
            direction /= np.linalg.norm(direction)
            candidates.append(x0 + radius * (null @ direction))
    def split_error(value):
        e1 = np.dot(value[:2], value[:2])
        return abs(e1 / max(target_norm2, 1.0e-30) - preferred_first_fraction)
    chosen = min(candidates, key=split_error)
    return b1 @ chosen[:2], b2 @ chosen[2:]


def reconstruct_post_state(v1, v2, w1, w2, u1, u2, ghat_post,
                           sampled_outcome, mass: float, moi: float,
                           aspect_ratio: float, rng: np.random.Generator) -> ReconstructionResult | None:
    v1, v2, w1, w2, u1, u2 = [np.asarray(x, dtype=float) for x in (v1, v2, w1, w2, u1, u2)]
    outcome = np.asarray(sampled_outcome, dtype=float)
    q, composition = float(outcome[0]), outcome[1:4]
    if not (0.0 <= q <= 1.0 + 1.0e-8) or np.any(composition < -1.0e-12):
        return None
    composition = composition / np.sum(composition)
    g = v2 - v1
    vcm = 0.5 * (v1 + v2)
    initial_energy = 0.5 * mass * np.dot(g, g) + moi * (np.dot(w1, w1) + np.dot(w2, w2))
    retained = q * initial_energy
    translational_energy = composition[0] * retained
    rotational_energy = (composition[1] + composition[2]) * retained
    gpost = np.sqrt(max(0.0, 2.0 * translational_energy / mass)) * np.asarray(ghat_post)
    frame = pair_frame(g / max(np.linalg.norm(g), 1.0e-30), u1, u2)
    normal = frame @ outcome[4:7]
    if np.isclose(aspect_ratio, 1.0):
        impulse = gpost - g
        separation = impulse / max(np.linalg.norm(impulse), 1.0e-30)
    else:
        lam, mu = outcome[10] * aspect_ratio, outcome[11] * aspect_ratio
        arm1 = lam * u1 + 0.5 * normal
        arm2 = mu * u2 - 0.5 * normal
        separation = arm1 - arm2
    initial_angular = 0.5 * mass * np.cross(separation, g) + moi * (w1 + w2)
    target_spin = initial_angular - 0.5 * mass * np.cross(separation, gpost)
    preferred = composition[1] / max(composition[1] + composition[2], 1.0e-30)
    solved = _solve_spins(target_spin, rotational_energy, preferred, u1, u2, moi, rng)
    if solved is None:
        return None
    w1post, w2post = solved
    v1post, v2post = vcm - 0.5 * gpost, vcm + 0.5 * gpost
    final_energy = 0.5 * mass * np.dot(gpost, gpost) + moi * (np.dot(w1post, w1post) + np.dot(w2post, w2post))
    final_angular = 0.5 * mass * np.cross(separation, gpost) + moi * (w1post + w2post)
    return ReconstructionResult(
        v1post, v2post, w1post, w2post,
        abs(final_energy - retained) / max(1.0, retained),
        np.linalg.norm(final_angular - initial_angular) / max(1.0, np.linalg.norm(initial_angular)),
    )

