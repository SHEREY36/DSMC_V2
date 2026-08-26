"""Joint nonparametric energy/geometry outcome library."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from dsmc_v2_contracts.io import OI, RunDataV2


def pair_frame(ghat: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Return a reproducible right-handed pair frame as matrix columns."""
    e0 = ghat / max(np.linalg.norm(ghat), 1.0e-30)
    trial = u1 - np.dot(u1, e0) * e0
    if np.linalg.norm(trial) < 1.0e-10:
        trial = u2 - np.dot(u2, e0) * e0
    if np.linalg.norm(trial) < 1.0e-10:
        candidate = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(candidate, e0)) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0])
        trial = candidate - np.dot(candidate, e0) * e0
    e1 = trial / np.linalg.norm(trial)
    return np.column_stack((e0, e1, np.cross(e0, e1)))


def conditioning_from_outcomes(run: RunDataV2) -> np.ndarray:
    values = np.asarray(run.outcomes["values"])
    c1, c2 = values[:, 0:3], values[:, 3:6]
    w1, w2 = values[:, 6:9], values[:, 9:12]
    u1, u2 = values[:, 12:15], values[:, 15:18]
    g = c2 - c1
    gmag = np.linalg.norm(g, axis=1)
    gh = g / np.maximum(gmag[:, None], 1.0e-30)
    total_w = np.einsum("ni,ni->n", w1, w1) + np.einsum("ni,ni->n", w2, w2)
    return np.column_stack((
        np.full(len(values), float(run.metadata["alpha"])),
        np.full(len(values), np.log(float(run.metadata["theta"]))),
        np.full(len(values), np.log(float(run.metadata["aspect_ratio"]))),
        np.log(np.maximum(gmag, 1.0e-12)), np.log1p(total_w),
        np.abs(np.einsum("ni,ni->n", gh, u1)),
        np.abs(np.einsum("ni,ni->n", gh, u2)),
        np.einsum("ni,ni->n", u1, u2) ** 2,
    ))


@dataclass
class EnergyLibrary:
    conditioning: np.ndarray
    outcomes: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    k_neighbors: int = 64
    _tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tree = cKDTree((self.conditioning - self.feature_mean) / self.feature_scale)

    def sample(self, conditioning: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        query = (np.asarray(conditioning, dtype=float) - self.feature_mean) / self.feature_scale
        k = min(self.k_neighbors, len(self.conditioning))
        distance, index = self._tree.query(query, k=k)
        distance, index = np.atleast_1d(distance), np.atleast_1d(index)
        weights = 1.0 / np.maximum(distance, 1.0e-8) ** 2
        chosen = int(rng.choice(index, p=weights / np.sum(weights)))
        return self.outcomes[chosen].copy()

    def save(self, path: str) -> None:
        np.savez_compressed(path, conditioning=self.conditioning, outcomes=self.outcomes,
                            feature_mean=self.feature_mean, feature_scale=self.feature_scale,
                            k_neighbors=np.array(self.k_neighbors), schema_version=np.array("2.0.0"))

    @classmethod
    def load(cls, path: str) -> "EnergyLibrary":
        data = np.load(path)
        if str(data["schema_version"]) != "2.0.0":
            raise ValueError("unsupported energy-library schema")
        return cls(data["conditioning"], data["outcomes"], data["feature_mean"],
                   data["feature_scale"], int(data["k_neighbors"]))


def build_energy_library(runs: list[RunDataV2], k_neighbors: int = 64) -> EnergyLibrary:
    if not runs:
        raise ValueError("at least one run is required")
    conditioning = np.concatenate([conditioning_from_outcomes(run) for run in runs])
    rows: list[np.ndarray] = []
    for run in runs:
        for row in np.asarray(run.outcomes["values"]):
            retained = row[OI["et_inelastic"]] + row[OI["er1_inelastic"]] + row[OI["er2_inelastic"]]
            q = retained / max(row[OI["e_initial"]], 1.0e-30)
            composition = np.array([row[OI["et_inelastic"]], row[OI["er1_inelastic"]],
                                    row[OI["er2_inelastic"]]]) / max(retained, 1.0e-30)
            gpre = np.array([row[OI[f"ghat_pre_{a}"]] for a in "xyz"])
            u1 = np.array([row[OI[f"u1_pre_{a}"]] for a in "xyz"])
            u2 = np.array([row[OI[f"u2_pre_{a}"]] for a in "xyz"])
            frame = pair_frame(gpre, u1, u2)
            normal = frame.T @ np.array([row[OI[f"contact_normal_{a}"]] for a in "xyz"])
            centerline = frame.T @ np.array([row[OI[f"centerline_{a}"]] for a in "xyz"])
            rows.append(np.concatenate((
                [q], composition, normal, centerline,
                [row[OI["contact_lambda"]], row[OI["contact_mu"]]],
            )))
    outcomes = np.asarray(rows)
    mean, scale = np.mean(conditioning, axis=0), np.std(conditioning, axis=0)
    scale[scale < 1.0e-10] = 1.0
    return EnergyLibrary(conditioning, outcomes, mean, scale, k_neighbors)

