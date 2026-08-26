"""Direction-only VSS rank-2 scattering sampler."""

from __future__ import annotations

import numpy as np

from coll_models_v2.vss import sample_vss_cosine


def sample_direction(ghat_pre: np.ndarray, alpha_eff: float,
                     rng: np.random.Generator) -> np.ndarray:
    ghat = np.asarray(ghat_pre, dtype=float)
    ghat /= max(np.linalg.norm(ghat), 1.0e-30)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, ghat)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, ghat) * ghat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ghat, e1)
    cosine = sample_vss_cosine(alpha_eff, rng)
    sine = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    azimuth = 2.0 * np.pi * rng.random()
    result = cosine * ghat + sine * (np.cos(azimuth) * e1 + np.sin(azimuth) * e2)
    return result / np.linalg.norm(result)

