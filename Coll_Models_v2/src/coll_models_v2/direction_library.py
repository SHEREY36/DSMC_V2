"""Rotationally equivariant, direction-only CTC transition library."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from dsmc_v2_contracts import load_run
from dsmc_v2_contracts.io import OI, RunDataV2, _vec


def pair_frame(relative_velocity: np.ndarray, axis1: np.ndarray,
               axis2: np.ndarray) -> np.ndarray:
    """Return a deterministic right-handed frame as three row vectors."""
    e1 = np.asarray(relative_velocity, dtype=float)
    e1 /= max(np.linalg.norm(e1), 1.0e-30)
    for candidate in (axis1, axis2, np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0])):
        e2 = np.asarray(candidate, dtype=float) - np.dot(candidate, e1) * e1
        norm = np.linalg.norm(e2)
        if norm > 1.0e-10:
            e2 /= norm
            break
    e3 = np.cross(e1, e2)
    return np.vstack((e1, e2, e3))


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(np.linalg.norm(vector), 1.0e-30)


def conditioning(alpha: float, theta: float, aspect_ratio: float,
                 c1: np.ndarray, c2: np.ndarray, w1: np.ndarray, w2: np.ndarray,
                 u1: np.ndarray, u2: np.ndarray, outgoing_fractions: np.ndarray,
                 mass: float, moi: float) -> tuple[np.ndarray, np.ndarray]:
    frame = pair_frame(c1 - c2, u1, u2)
    vcm = 0.5 * (c1 + c2)
    etr = mass * (np.dot(c1 - vcm, c1 - vcm) + np.dot(c2 - vcm, c2 - vcm))
    er1, er2 = moi * np.dot(w1, w1), moi * np.dot(w2, w2)
    total = max(etr + er1 + er2, 1.0e-30)
    er = er1 + er2
    values = [alpha, np.log(theta), np.log(aspect_ratio), etr / total,
              er1 / max(er, 1.0e-30), *np.asarray(outgoing_fractions, dtype=float)]
    for vector in (u1, u2, _unit(w1), _unit(w2)):
        values.extend(frame @ vector)
    return np.asarray(values), frame


@dataclass
class DirectionLibrary:
    mean: np.ndarray
    scale: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    features: np.ndarray
    directions: np.ndarray
    coverage_radius_95: float

    def __post_init__(self) -> None:
        self.tree = cKDTree(self.features)

    def save(self, path: str) -> None:
        np.savez_compressed(path, schema_version=np.array("2.1.0"), mean=self.mean,
                            scale=self.scale, lower=self.lower, upper=self.upper,
                            features=self.features,
                            directions=self.directions,
                            coverage_radius_95=self.coverage_radius_95)

    @classmethod
    def load(cls, path: str) -> "DirectionLibrary":
        data = np.load(path, allow_pickle=False)
        if str(data["schema_version"]) != "2.1.0":
            raise ValueError("unsupported rotational-direction schema")
        return cls(data["mean"], data["scale"], data["lower"], data["upper"],
                   data["features"],
                   data["directions"], float(data["coverage_radius_95"]))

    def select(self, raw_query: np.ndarray, rng: np.random.Generator,
               neighbours: int = 64) -> np.ndarray:
        raw_query = np.asarray(raw_query)
        # The first three entries are alpha, log(theta), and log(AR). Other
        # features are instantaneous states and are handled by nearest donors.
        if np.any(raw_query[:3] < self.lower[:3] - 1.0e-12) \
                or np.any(raw_query[:3] > self.upper[:3] + 1.0e-12):
            raise ValueError("rotational-direction query lies outside the CTC design hull")
        query = (raw_query - self.mean) / self.scale
        k = min(int(neighbours), len(self.features))
        distance, index = self.tree.query(query, k=k)
        distance, index = np.atleast_1d(distance), np.atleast_1d(index)
        weights = 1.0 / np.maximum(distance, 1.0e-10)
        weights /= np.sum(weights)
        return self.directions[int(rng.choice(index, p=weights))]


def _build_direction_library(runs, run_count: int,
                             maximum_donors: int = 500_000) -> DirectionLibrary:
    """Build a deterministic, node-stratified donor subset.

    The full accepted streams remain the source of routing/VSS statistics, but
    loading tens of millions of donors at DSMC runtime is unnecessary.  Every
    run receives the same bounded quota and is sampled uniformly by record
    index, keeping the deployed cKDTree below a documented memory ceiling.
    """
    features, directions = [], []
    quota = max(64, int(maximum_donors) // max(run_count, 1))
    for run in runs:
        values = run.outcomes["values"]
        mass = float(run.metadata["mass"])
        moi = float(run.metadata["moi_perpendicular"])
        indices = np.arange(len(values)) if len(values) <= quota else np.linspace(
            0, len(values) - 1, quota, dtype=int)
        for index in indices:
            row = values[index]
            c1, c2 = (np.array([row[OI[f"c{p}_pre_{a}"]] for a in "xyz"])
                      for p in (1, 2))
            w1, w2 = (np.array([row[OI[f"omega{p}_pre_{a}"]] for a in "xyz"])
                      for p in (1, 2))
            u1, u2 = (np.array([row[OI[f"u{p}_pre_{a}"]] for a in "xyz"])
                      for p in (1, 2))
            wp1, wp2 = (np.array([row[OI[f"omega{p}_post_{a}"]] for a in "xyz"])
                        for p in (1, 2))
            et = row[OI["et_inelastic"]]
            er1, er2 = row[OI["er1_inelastic"]], row[OI["er2_inelastic"]]
            total = max(et + er1 + er2, 1.0e-30)
            outgoing = np.array([et / total, er1 / max(er1 + er2, 1.0e-30)])
            feature, frame = conditioning(
                float(run.metadata["alpha"]), float(run.metadata["theta"]),
                float(run.metadata["aspect_ratio"]), c1, c2, w1, w2, u1, u2,
                outgoing, mass, moi)
            if min(np.linalg.norm(wp1), np.linalg.norm(wp2)) <= 1.0e-12:
                continue
            features.append(feature)
            directions.append(np.concatenate((frame @ _unit(wp1), frame @ _unit(wp2))))
    if len(features) < 64:
        raise ValueError("at least 64 nondegenerate CTC outcomes are required")
    raw = np.asarray(features)
    mean, scale = np.mean(raw, axis=0), np.std(raw, axis=0)
    scale[scale < 1.0e-10] = 1.0
    standardized = (raw - mean) / scale
    tree = cKDTree(standardized)
    distance, _ = tree.query(standardized, k=2)
    return DirectionLibrary(mean, scale, np.min(raw, axis=0), np.max(raw, axis=0),
                            standardized, np.asarray(directions),
                            float(np.quantile(distance[:, 1], 0.95)))


def build_direction_library(runs: list[RunDataV2],
                            maximum_donors: int = 500_000) -> DirectionLibrary:
    return _build_direction_library(runs, len(runs), maximum_donors)


def build_direction_library_from_paths(run_directories,
                                       maximum_donors: int = 500_000) -> DirectionLibrary:
    """Load one raw shard at a time so artifact export has bounded open files."""
    paths = list(run_directories)

    def runs():
        for path in paths:
            yield load_run(path)

    return _build_direction_library(runs(), len(paths), maximum_donors)
