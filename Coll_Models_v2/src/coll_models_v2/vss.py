"""VSS angular kernel fitted only to the CTC direction-only rank-2 moment."""

from __future__ import annotations

import math

import numpy as np


B2_MAX = 18.0 - 12.0 * math.sqrt(2.0)


def legendre(order: int, cosine: np.ndarray | float) -> np.ndarray:
    x = np.asarray(cosine, dtype=float)
    if order == 1:
        return x
    if order == 2:
        return 0.5 * (3.0 * x * x - 1.0)
    if order == 3:
        return 0.5 * (5.0 * x**3 - 3.0 * x)
    if order == 4:
        return (35.0 * x**4 - 30.0 * x * x + 3.0) / 8.0
    raise ValueError("only Legendre orders 1 through 4 are supported")


def vss_rank2_moment(alpha_eff: float) -> float:
    alpha_eff = float(alpha_eff)
    if not np.isfinite(alpha_eff) or alpha_eff <= 0.0:
        raise ValueError("alpha_eff must be finite and positive")
    return 6.0 * alpha_eff / ((alpha_eff + 1.0) * (alpha_eff + 2.0))


def alpha_eff_from_b2(b2: float) -> float:
    """Return the larger (forward) VSS root; never clamp an invalid target."""
    b2 = float(b2)
    if not np.isfinite(b2) or b2 <= 0.0 or b2 > B2_MAX + 1.0e-12:
        raise ValueError(f"B2={b2!r} lies outside the forward-VSS range (0,{B2_MAX:.12g}]")
    discriminant = b2 * b2 - 36.0 * b2 + 36.0
    return float((6.0 - 3.0 * b2 + math.sqrt(max(0.0, discriminant))) / (2.0 * b2))


def sample_vss_cosine(alpha_eff: float, rng: np.random.Generator) -> float:
    alpha_eff = float(alpha_eff)
    if alpha_eff <= 0.0:
        raise ValueError("alpha_eff must be positive")
    return float(2.0 * rng.random() ** (1.0 / alpha_eff) - 1.0)


def angular_diagnostics(ghat_pre: np.ndarray, ghat_post: np.ndarray) -> dict:
    ghat_pre = np.asarray(ghat_pre, dtype=float)
    ghat_post = np.asarray(ghat_post, dtype=float)
    cosine = np.clip(np.einsum("ni,ni->n", ghat_pre, ghat_post), -1.0, 1.0)
    b2 = float(np.mean(1.0 - legendre(2, cosine)))
    result = {"n": int(len(cosine)), "B2": b2,
              "alpha_eff": alpha_eff_from_b2(b2)}
    for order in range(1, 5):
        result[f"mean_P{order}"] = float(np.mean(legendre(order, cosine)))
    return result

