"""Streaming non-Gaussian HCS diagnostics in collision-count time."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


MOMENT_FIELDS = (
    "time", "tau", "Ttr", "Trot", "theta", "c2", "c4", "c6",
    "w2", "w4", "w6", "c2w2", "a20", "a02", "a11", "A_cu",
    "A_cw_quadrupolar",
)


def reduced_observables(velocity: np.ndarray, omega: np.ndarray,
                        axis: np.ndarray, mass: float, inertia: float,
                        sphere: bool = False) -> dict[str, float | np.ndarray | None]:
    """Return reduced moments for three translational and two rotational DOF."""
    velocity = np.asarray(velocity, dtype=float)
    omega = np.asarray(omega, dtype=float)
    axis = np.asarray(axis, dtype=float)
    peculiar = velocity - np.mean(velocity, axis=0)
    ttr = float(mass * np.einsum("ni,ni->", peculiar, peculiar)
                / (3.0 * len(peculiar)))
    if not np.isfinite(ttr) or ttr <= 0.0:
        raise ValueError("non-Gaussian sampling requires positive Ttr")
    c = peculiar / np.sqrt(2.0 * ttr / float(mass))
    c2_values = np.einsum("ni,ni->n", c, c)
    result: dict[str, float | np.ndarray | None] = {
        "Ttr": ttr,
        "c": np.sqrt(c2_values),
        "c2_values": c2_values,
        "c2": float(np.mean(c2_values)),
        "c4": float(np.mean(c2_values**2)),
        "c6": float(np.mean(c2_values**3)),
    }
    result["a20"] = 4.0 * float(result["c4"]) / 15.0 - 1.0
    if sphere:
        result.update(Trot=None, theta=None, w=None, w2_values=None, x_values=None,
                      w2=None, w4=None, w6=None, c2w2=None, a02=None,
                      a11=None, A_cu=None, A_cw_quadrupolar=None)
        return result

    trot = float(inertia * np.einsum("ni,ni->", omega, omega)
                 / (2.0 * len(omega)))
    if not np.isfinite(trot) or trot <= 0.0:
        raise ValueError("non-Gaussian sampling requires positive Trot")
    w = omega / np.sqrt(2.0 * trot / float(inertia))
    w2_values = np.einsum("ni,ni->n", w, w)
    x_values = c2_values * w2_values
    c2w2 = float(np.mean(x_values))
    axes = axis / np.linalg.norm(axis, axis=1)[:, None]
    acu = np.einsum("ni,ni->n", c, axes)**2 - c2_values / 3.0
    cw = np.einsum("ni,ni->n", c, w)
    result.update(
        Trot=trot, theta=ttr / trot, w=np.sqrt(w2_values),
        w2_values=w2_values, x_values=x_values,
        w2=float(np.mean(w2_values)), w4=float(np.mean(w2_values**2)),
        w6=float(np.mean(w2_values**3)), c2w2=c2w2,
        a02=0.5 * float(np.mean(w2_values**2)) - 1.0,
        a11=(2.0 / 3.0) * c2w2 - 1.0,
        A_cu=float(np.mean(acu)),
        A_cw_quadrupolar=float(np.mean(cw**2 - x_values / 3.0)),
    )
    return result


class NonGaussianDiagnostics:
    """Accumulate moments and histogram counts without storing particle dumps."""

    def __init__(self, config: dict, output_path: str | Path, particle_count: int,
                 mass: float, inertia: float, sphere: bool = False):
        cfg = config.get("diagnostics", {}).get("non_gaussian", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.start = float(cfg.get("sample_start_tau", 500.0))
        self.end = float(cfg.get("sample_end_tau", 1500.0))
        self.delta = float(cfg.get("sample_delta_tau", 5.0))
        if self.delta <= 0.0 or self.end < self.start:
            raise ValueError("invalid non-Gaussian collision-time sampling window")
        self.next_tau = self.start
        self.expected_samples = int(np.floor((self.end - self.start) / self.delta
                                             + 1.0e-10)) + 1
        self.output_path = Path(output_path)
        self.moment_path = self.output_path.with_name(
            self.output_path.stem + "_ng_moments.csv")
        self.histogram_path = self.output_path.with_name(
            self.output_path.stem + "_ng_histograms.npz")
        self.summary_path = self.output_path.with_name(
            self.output_path.stem + "_ng_summary.json")
        self.particle_count = int(particle_count)
        self.mass, self.inertia, self.sphere = float(mass), float(inertia), bool(sphere)
        self.sample_count = 0
        self.sums = {name: 0.0 for name in MOMENT_FIELDS[2:]}
        self.counts = {name: 0 for name in MOMENT_FIELDS[2:]}
        self.edges = {
            "c": np.linspace(0.0, float(cfg.get("c_max", 8.0)), 257),
            "w": np.geomspace(float(cfg.get("w_min", 1.0e-3)),
                               float(cfg.get("w_max", 64.0)), 257),
            "x": np.geomspace(float(cfg.get("x_min", 1.0e-6)),
                               float(cfg.get("x_max", 1.0e5)), 257),
        }
        self.hist = {name: np.zeros(256, dtype=np.int64) for name in self.edges}
        self.underflow = {name: 0 for name in self.edges}
        self.overflow = {name: 0 for name in self.edges}
        self.tail_thresholds = {
            "c": float(cfg.get("c_tail_threshold", 4.0)),
            "w": float(cfg.get("w_tail_threshold", 6.0)),
            "x": float(cfg.get("x_tail_threshold", 50.0)),
        }
        self.tail_counts = {name: 0 for name in self.edges}
        self.minimum_tail_count = int(cfg.get("minimum_tail_count", 1000))
        self._handle = self._writer = None
        if self.enabled:
            self.moment_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.moment_path.open("w", newline="", buffering=65536)
            self._writer = csv.DictWriter(self._handle, fieldnames=MOMENT_FIELDS)
            self._writer.writeheader()

    def maybe_sample(self, time: float, tau: float, state) -> bool:
        if not self.enabled or tau + 1.0e-10 < self.next_tau or self.next_tau > self.end + 1e-10:
            return False
        values = reduced_observables(state.velocity, state.omega, state.axis,
                                     self.mass, self.inertia, self.sphere)
        row = {"time": float(time), "tau": float(tau)}
        for name in MOMENT_FIELDS[2:]:
            value = values.get(name)
            row[name] = "" if value is None else float(value)
            if value is not None and np.isfinite(value):
                self.sums[name] += float(value)
                self.counts[name] += 1
        self._writer.writerow(row)
        self._handle.flush()
        for name, value_name in (("c", "c"), ("w", "w"), ("x", "x_values")):
            samples = values.get(value_name)
            if samples is None:
                continue
            samples = np.asarray(samples, dtype=float)
            self.hist[name] += np.histogram(samples, bins=self.edges[name])[0]
            self.underflow[name] += int(np.sum(samples < self.edges[name][0]))
            self.overflow[name] += int(np.sum(samples >= self.edges[name][-1]))
            self.tail_counts[name] += int(np.sum(samples >= self.tail_thresholds[name]))
        self.sample_count += 1
        while self.next_tau <= tau + 1.0e-10:
            self.next_tau += self.delta
        return True

    def close(self) -> dict | None:
        if not self.enabled:
            return None
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        with self.histogram_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                **{f"{name}_edges": self.edges[name] for name in self.edges},
                **{f"{name}_counts": self.hist[name] for name in self.hist},
                **{f"{name}_underflow": np.array(self.underflow[name]) for name in self.edges},
                **{f"{name}_overflow": np.array(self.overflow[name]) for name in self.edges},
            )
        means = {name: (self.sums[name] / self.counts[name]
                        if self.counts[name] else None)
                 for name in self.sums}
        summary = {
            "sampling_complete": self.sample_count == self.expected_samples,
            "sample_start_tau": self.start, "sample_end_tau": self.end,
            "sample_delta_tau": self.delta,
            "expected_samples": self.expected_samples,
            "n_samples": self.sample_count,
            "n_particle_samples": self.sample_count * self.particle_count,
            "means": means,
            "tail_thresholds": self.tail_thresholds,
            "tail_counts": self.tail_counts,
            "minimum_tail_count": self.minimum_tail_count,
            "tail_fit_ready": {
                name: count >= self.minimum_tail_count
                for name, count in self.tail_counts.items()
            },
            "moments_file": str(self.moment_path),
            "histograms_file": str(self.histogram_path),
        }
        temporary = self.summary_path.with_suffix(self.summary_path.suffix + ".tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.summary_path)
        return summary
