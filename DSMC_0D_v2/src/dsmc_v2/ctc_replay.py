"""Collision-flux reservoirs for exact CTC relabelling of DSMC USF states.

The sampler observes pairs after NTC relative-speed screening and before the
orientation-dependent acceptance or collision closure.  Marginalising the
random NTC normal gives the required ``g f_1 f_2`` incoming-pair measure.  It
therefore must not observe only closure-accepted pairs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import FEATURE_NAMES, cell_features_with_domain


REPLAY_RECORD_FIELDS = (
    *(f"c1_{axis}" for axis in "xyz"), *(f"c2_{axis}" for axis in "xyz"),
    *(f"omega1_{axis}" for axis in "xyz"),
    *(f"omega2_{axis}" for axis in "xyz"),
    *(f"u1_{axis}" for axis in "xyz"), *(f"u2_{axis}" for axis in "xyz"),
)
REPLAY_WIDTH = len(REPLAY_RECORD_FIELDS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _systematic_indices(weight: np.ndarray, count: int, seed: int) -> np.ndarray:
    weight = np.asarray(weight, dtype=float)
    weight /= np.sum(weight)
    positions = (np.random.default_rng(seed).random() + np.arange(count)) / count
    return np.searchsorted(np.cumsum(weight), positions, side="right").clip(
        0, len(weight) - 1)


@dataclass(frozen=True)
class ReplayWindow:
    start_tau: float
    end_tau: float

    def contains(self, tau: float) -> bool:
        return self.start_tau <= tau < self.end_tau


class CollisionFluxReservoir:
    """Fixed-memory reservoirs of post-NTC incoming pairs in several windows."""

    def __init__(self, output_prefix: str | Path, windows: list[tuple[float, float]],
                 capacity: int, seed: int, *, alpha: float, aspect_ratio: float,
                 mass: float, inertia: float,
                 source_provenance: dict | None = None):
        if capacity < 100:
            raise ValueError("replay reservoir capacity must be at least 100")
        parsed = [ReplayWindow(float(start), float(end)) for start, end in windows]
        if not parsed or any(window.start_tau < 0.0
                             or window.end_tau <= window.start_tau
                             for window in parsed):
            raise ValueError("replay windows must be non-empty increasing intervals")
        ordered = sorted(parsed, key=lambda window: window.start_tau)
        if any(left.end_tau > right.start_tau for left, right in zip(ordered, ordered[1:])):
            raise ValueError("replay windows must not overlap")
        self.output_prefix = Path(output_prefix)
        self.windows = ordered
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.aspect_ratio = float(aspect_ratio)
        self.mass = float(mass)
        self.inertia = float(inertia)
        self.source_provenance = dict(source_provenance or {})
        self.rng = [np.random.default_rng(int(seed) + 7919 * (i + 1))
                    for i in range(len(ordered))]
        self.records = [np.empty((self.capacity, REPLAY_WIDTH), dtype="<f8")
                        for _ in ordered]
        # Records are normalized state by state before replay, so their stored
        # relative speed is not the speed that generated the NTC selection.
        # Retain that physical selection speed separately for the exact 1/g
        # conversion back to the cell measure.
        self.selection_speed = [np.empty(self.capacity, dtype=float)
                                for _ in ordered]
        self.seen = np.zeros(len(ordered), dtype=np.int64)
        self.stored = np.zeros(len(ordered), dtype=np.int64)

    def observe(self, *, tau: float, v1: np.ndarray, v2: np.ndarray,
                omega1: np.ndarray, omega2: np.ndarray,
                axis1: np.ndarray, axis2: np.ndarray,
                mean_velocity: np.ndarray, trot: float) -> None:
        """Observe one screened pair, normalized to ``Trot=1``.

        A common velocity/spin scale preserves every dimensionless invariant
        and the instantaneous temperature ratio while keeping the CTC contact
        integration at the scale used by its restitution calibration.
        """
        index = next((i for i, window in enumerate(self.windows)
                      if window.contains(float(tau))), None)
        if index is None:
            return
        if not np.isfinite(trot) or trot <= 0.0:
            raise ValueError("cannot replay a state with non-positive Trot")
        self.seen[index] += 1
        seen = int(self.seen[index])
        if seen <= self.capacity:
            destination = seen - 1
            self.stored[index] = seen
        else:
            destination = int(self.rng[index].integers(seen))
            if destination >= self.capacity:
                return
        scale = 1.0 / np.sqrt(float(trot))
        record = self.records[index][destination]
        record[0:3] = (np.asarray(v1) - mean_velocity) * scale
        record[3:6] = (np.asarray(v2) - mean_velocity) * scale
        record[6:9] = np.asarray(omega1) * scale
        record[9:12] = np.asarray(omega2) * scale
        record[12:15] = axis1
        record[15:18] = axis2
        self.selection_speed[index][destination] = float(np.linalg.norm(
            np.asarray(v1) - np.asarray(v2)))

    def diagnostics(self) -> dict:
        return {
            "sampling_contract": "post_ntc_pre_orientation_collision_flux_v1",
            "capacity_per_window": self.capacity,
            "windows": [
                {"start_tau": window.start_tau, "end_tau": window.end_tau,
                 "n_seen": int(self.seen[i]), "n_stored": int(self.stored[i])}
                for i, window in enumerate(self.windows)
            ],
        }

    def _measure_cell(self, records: np.ndarray, selection_speed: np.ndarray,
                      seed: int) -> dict:
        c1, c2 = records[:, 0:3], records[:, 3:6]
        w1, w2 = records[:, 6:9], records[:, 9:12]
        u1, u2 = records[:, 12:15], records[:, 15:18]
        selection_speed = np.asarray(selection_speed, dtype=float)
        if selection_speed.shape != (len(records),) \
                or np.any(~np.isfinite(selection_speed)) \
                or np.any(selection_speed <= 1.0e-14):
            raise ValueError("replay reservoir contains a zero-relative-speed pair")
        # The reservoir is collision-flux weighted.  One factor 1/g recovers
        # the cell measure before evaluating its one-particle/U invariants.
        velocity = np.vstack((c1, c2))
        omega = np.vstack((w1, w2))
        axis = np.vstack((u1, u2))
        inverse_speed = np.concatenate((1.0 / selection_speed,
                                        1.0 / selection_speed))
        particle_weight = inverse_speed / np.sum(inverse_speed)

        # Temperatures are simple one-particle moments, so evaluate them with
        # the exact self-normalised 1/g importance weights.  Do not introduce
        # an avoidable resampling error into the theta passed back to CTC.
        mean_velocity = np.sum(particle_weight[:, None] * velocity, axis=0)
        peculiar = velocity - mean_velocity
        ttr = self.mass * np.sum(
            particle_weight * np.einsum("ni,ni->n", peculiar, peculiar)) / 3.0
        trot = self.inertia * np.sum(
            particle_weight * np.einsum("ni,ni->n", omega, omega)) / 2.0

        # The fourteen cell invariants contain U-statistics.  A deterministic
        # systematic resample is the bounded-memory representation of the
        # de-biased cell measure used by the existing feature implementation.
        pick = _systematic_indices(inverse_speed, len(velocity), seed)
        feature_velocity = velocity[pick]
        feature_omega = omega[pick]
        feature_axis = axis[pick]
        features, domain = cell_features_with_domain(
            feature_velocity, feature_omega, feature_axis,
            self.mass, self.inertia, sphere=False)
        axis_norm_error = float(np.max(np.abs(
            np.linalg.norm(axis, axis=1) - 1.0)))
        omega_axis_error = float(np.max(
            np.abs(np.einsum("ni,ni->n", omega, axis))
            / np.maximum(1.0, np.linalg.norm(omega, axis=1))))
        return {
            "theta": float(ttr / trot),
            "temperature_translational": float(ttr),
            "temperature_rotational": float(trot),
            "cell_features": dict(zip(FEATURE_NAMES, features.tolist())),
            "domain_features": dict(zip(FEATURE_NAMES, domain.tolist())),
            "cell_measure_effective_particles": int(len(velocity)),
            "temperature_estimator": "self_normalized_inverse_speed_v1",
            "feature_estimator": "inverse_speed_systematic_resample_v1",
            "maximum_axis_norm_error": axis_norm_error,
            "maximum_omega_axis_perpendicularity_error": omega_axis_error,
        }

    def finalize(self, minimum_records: int | None = None) -> list[dict]:
        minimum = self.capacity if minimum_records is None else int(minimum_records)
        if minimum < 100 or minimum > self.capacity:
            raise ValueError("minimum replay records must lie in [100, capacity]")
        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        outputs = []
        for index, window in enumerate(self.windows):
            count = int(self.stored[index])
            if count < minimum:
                raise RuntimeError(
                    f"replay window {index} stored {count} records; need {minimum}")
            records = np.ascontiguousarray(self.records[index][:count], dtype="<f8")
            if not np.all(np.isfinite(records)):
                raise ValueError(f"replay window {index} contains NaN or Inf")
            stem = Path(f"{self.output_prefix}_window_{index:02d}")
            # ``Path.with_suffix`` is not safe here: decimal points in names
            # such as ``AR_1.500_alpha_0.800`` are interpreted as a suffix and
            # would collapse every branch/window to ``AR_1.500_alpha_0.bin``.
            binary = Path(f"{stem}.bin")
            temporary = Path(f"{binary}.tmp")
            records.tofile(temporary)
            os.replace(temporary, binary)
            measured = self._measure_cell(
                records, self.selection_speed[index][:count],
                seed=0xCE110 + index)
            replay_qa = {
                "record_count_pass": bool(count >= minimum),
                "axis_norm_pass": bool(measured["maximum_axis_norm_error"] <= 2.0e-6),
                "omega_axis_perpendicularity_pass": bool(
                    measured["maximum_omega_axis_perpendicularity_error"] <= 2.0e-6),
                # Each accepted pair was divided by the instantaneous
                # sqrt(Trot).  This aggregate check catches a wrong inertia,
                # scale, or measure conversion without demanding an
                # impossible exact equality from a finite reservoir.
                "normalised_trot_pass": bool(abs(
                    measured["temperature_rotational"] - 1.0) <= 0.03),
            }
            replay_qa["pass"] = bool(all(replay_qa.values()))
            payload = {
                "schema_version": "dsmc-ctc-replay-v1",
                "record_dtype": "little_endian_float64",
                "record_width": REPLAY_WIDTH,
                "record_fields": list(REPLAY_RECORD_FIELDS),
                "record_count": count,
                "source_pairs_seen": int(self.seen[index]),
                "reservoir_capacity": self.capacity,
                "selection_measure": "post_ntc_pre_orientation_collision_flux",
                "cell_measure_debiasing": "inverse_physical_selection_speed_v1",
                "normalization": "instantaneous_common_scale_to_Trot_1",
                "alpha": self.alpha,
                "aspect_ratio": self.aspect_ratio,
                "mass": self.mass,
                "moi_perpendicular": self.inertia,
                "tau_window": [window.start_tau, window.end_tau],
                "window_index": index,
                "source_provenance": self.source_provenance,
                "binary_file": str(binary),
                "binary_sha256": _sha256(binary),
                "sampling_contract": "post_ntc_pre_orientation_collision_flux_v1",
                "qa": replay_qa,
                **measured,
            }
            metadata = Path(f"{stem}.json")
            metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            outputs.append({"binary": str(binary), "metadata": str(metadata), **payload})
        return outputs
