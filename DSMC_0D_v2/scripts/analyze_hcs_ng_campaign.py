#!/usr/bin/env python3
"""Aggregate HCS non-Gaussian replicates with time-correlation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


OBSERVABLES = (
    "theta_tr_over_rot", "theta_rot_over_tr",
    "a20", "a02", "a11", "A_cu", "A_cw_quadrupolar",
)
LOG_THETA = "log_theta_tr_over_rot"
ANALYSIS_OBSERVABLES = OBSERVABLES + (LOG_THETA,)
# Report both temperature-ratio conventions, but never gate them twice with
# the same absolute tolerance.  log(theta) changes only sign under reciprocal
# convention, so its absolute difference is coordinate-invariant.
CONTROL_OBSERVABLES = (
    LOG_THETA, "a20", "a02", "a11", "A_cu", "A_cw_quadrupolar",
)
STATIONARITY_OBSERVABLES = (
    LOG_THETA, "a20", "a02", "a11", "A_cu", "A_cw_quadrupolar",
)
MAX_MAJORANT_VIOLATIONS_PER_ACCEPTED_PAIR = 1.0e-5
MAXIMUM_BULK_TO_THERMAL_TEMPERATURE_RATIO = 1.0e-12
MAXIMUM_EVALUATION_CORRECTION_FALLBACK_FRACTION = 1.0e-2
MINIMUM_THETA_HULL_LOG_MARGIN = 0.02
MINIMUM_STATIONARITY_SAMPLES = 15
STATIONARITY_ABSOLUTE_TOLERANCE = {
    LOG_THETA: 0.03,
    "a20": 0.01,
    "a02": 0.01,
    "a11": 0.01,
    "A_cu": 0.01,
    "A_cw_quadrupolar": 0.01,
}
STATIONARITY_PRECISION_LIMIT = {
    LOG_THETA: 0.06,
    "a20": 0.04,
    "a02": 0.04,
    "a11": 0.04,
    "A_cu": 0.04,
    "A_cw_quadrupolar": 0.04,
}


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4 or np.var(values) == 0.0:
        return 1.0
    centered = values - np.mean(values)
    corr = np.correlate(centered, centered, mode="full")[len(values)-1:]
    corr /= corr[0]
    total = 1.0
    for value in corr[1:len(values)//2]:
        if value <= 0.0:
            break
        total += 2.0 * float(value)
    return max(1.0, total)


def bootstrap_ci(values: list[float], seed: int = 260918) -> list[float] | None:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return None
    if len(clean) == 1:
        return [float(clean[0]), float(clean[0])]
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(clean, size=(4000, len(clean)), replace=True), axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def load_series(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {"tau": np.asarray([
        float(row["tau"]) if row.get("tau") not in ("", None) else np.nan
        for row in rows], dtype=float)}
    for name in OBSERVABLES:
        values = []
        for row in rows:
            raw = row.get(name, "")
            # Protocol-v1 files used the ambiguous name ``theta`` for the
            # closure convention Ttr/Trot.  Read them for forensic analysis,
            # but every new file writes both conventions explicitly.
            if raw in ("", None) and name == "theta_tr_over_rot":
                raw = row.get("theta", "")
            if raw in ("", None) and name == "theta_rot_over_tr":
                legacy = row.get("theta", "")
                raw = "" if legacy in ("", None) else 1.0 / float(legacy)
            values.append(float(raw) if raw not in ("", None) else np.nan)
        result[name] = np.asarray(values, dtype=float)
    theta = result["theta_tr_over_rot"]
    result[LOG_THETA] = np.where(theta > 0.0, np.log(theta), np.nan)
    return result


def block_stationarity(values: np.ndarray, name: str) -> dict:
    """Detect late drift/change points on an ensemble-mean sampled curve."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    tolerance = float(STATIONARITY_ABSOLUTE_TOLERANCE[name])
    base = {"pass": False, "absolute_tolerance": tolerance,
            "n_samples": int(len(clean))}
    if len(clean) < MINIMUM_STATIONARITY_SAMPLES:
        base["reason"] = "too_few_late_samples"
        return base
    blocks = [part for part in np.array_split(clean, 5) if len(part)]
    means = np.asarray([np.mean(part) for part in blocks], dtype=float)
    x = np.arange(len(clean), dtype=float)
    slope = float(np.polyfit(x, clean, 1)[0])
    linear_change = slope * max(len(clean) - 1, 1)
    first_last = float(means[-1] - means[0])
    maximum_adjacent = float(np.max(np.abs(np.diff(means))))
    block_range = float(np.ptp(means))
    passed = (abs(first_last) <= tolerance
              and abs(linear_change) <= tolerance
              and maximum_adjacent <= tolerance
              and block_range <= 1.5 * tolerance)
    base.update({
        "pass": bool(passed),
        "block_means": means.tolist(),
        "first_last_change": first_last,
        "linear_change_over_window": linear_change,
        "maximum_adjacent_block_change": maximum_adjacent,
        "block_mean_range": block_range,
        "reason": None if passed else "late_window_change_detected",
    })
    return base


def replicate_stationarity(curves: list[np.ndarray], name: str) -> dict:
    """Test a late-window drift against independent-realization noise.

    A hard change-point tolerance on one N=2000 ensemble curve rejected the
    exact elastic equilibrium control in protocol v5.  The physical question
    is instead whether the early-to-late change is resolved across independent
    realizations.  A large uncertainty cannot manufacture a pass: the three-SE
    resolution must also remain below a declared ceiling.
    """
    finite = [np.asarray(curve, dtype=float)[np.isfinite(curve)] for curve in curves]
    tolerance = float(STATIONARITY_ABSOLUTE_TOLERANCE[name])
    precision_limit = float(STATIONARITY_PRECISION_LIMIT[name])
    base = {"pass": False, "absolute_tolerance": tolerance,
            "precision_limit": precision_limit,
            "n_replicates": len(finite),
            "n_samples": min(map(len, finite), default=0)}
    if len(finite) < 2:
        base["reason"] = "two_independent_replicates_required"
        return base
    common = min(map(len, finite))
    if common < MINIMUM_STATIONARITY_SAMPLES:
        base["reason"] = "too_few_late_samples"
        return base
    aligned = [curve[-common:] for curve in finite]
    third = max(1, common // 3)
    drifts = np.asarray([
        float(np.mean(curve[-third:]) - np.mean(curve[:third]))
        for curve in aligned], dtype=float)
    mean_drift = float(np.mean(drifts))
    stderr = float(np.std(drifts, ddof=1) / np.sqrt(len(drifts)))
    resolution = 3.0 * stderr
    statistical_tolerance = max(tolerance, resolution)
    consistency_pass = abs(mean_drift) <= statistical_tolerance
    precision_pass = resolution <= precision_limit
    ensemble = np.mean(np.stack(aligned), axis=0)
    blocks = [part for part in np.array_split(ensemble, 5) if len(part)]
    block_means = np.asarray([np.mean(part) for part in blocks], dtype=float)
    base.update({
        "pass": bool(consistency_pass and precision_pass),
        "replicate_drifts": drifts.tolist(),
        "replicate_mean_drift": mean_drift,
        "replicate_drift_stderr": stderr,
        "statistical_resolution_3se": resolution,
        "statistical_tolerance": statistical_tolerance,
        "statistical_consistency_pass": bool(consistency_pass),
        "precision_pass": bool(precision_pass),
        "block_means_diagnostic_only": block_means.tolist(),
        "reason": (None if consistency_pass and precision_pass else
                   "late_window_drift_resolved" if not consistency_pass else
                   "late_window_drift_under_resolved"),
    })
    return base


def replicate_record(row: dict[str, str]) -> dict:
    prefix = Path(row["output_prefix"])
    result_path = Path(str(prefix) + ".json")
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text())
    if result.get("run_status", "complete") != "complete":
        raise RuntimeError(f"task {row['task_id']} result is not complete")
    summary = result.get("non_gaussian") or {}
    series = load_series(Path(str(prefix) + "_ng_moments.csv"))
    record = {
        "task_id": int(row["task_id"]), "result": result,
        "replicate": int(row["replicate"]), "seed": int(row["seed"]),
        "initial_theta": float(row.get("initial_theta") or 1.0),
        "required_dissipation_horizon": float(
            row.get("dissipation_horizon") or 0.0),
        "protocol_version": row.get("protocol_version", "hcs-ng-v1"),
        "model_variant": row.get("model_variant", "baseline"),
        "invariant_corrections": row.get("invariant_corrections", "false") == "true",
        "sampling_complete": bool(summary.get("sampling_complete", False)),
        "tail_counts": summary.get("tail_counts", {}),
        "tail_thresholds": summary.get("tail_thresholds", {}),
        "histograms_file": summary.get("histograms_file"),
        "minimum_tail_count": int(summary.get("minimum_tail_count", 1000)),
        "observables": {}, "series": series,
    }
    for name, values in series.items():
        if name == "tau":
            continue
        clean = values[np.isfinite(values)]
        if not len(clean):
            record["observables"][name] = None; continue
        third = max(1, len(clean) // 3)
        drift = abs(float(np.mean(clean[-third:]) - np.mean(clean[:third])))
        scale = max(abs(float(np.mean(clean))),
                    1e-12 if name.startswith("theta_") else 0.05)
        tau_int = integrated_autocorrelation_time(clean)
        record["observables"][name] = {
            "mean": float(np.mean(clean)),
            "signed_early_late_drift": float(np.mean(clean[-third:])
                                             - np.mean(clean[:third])),
            "drift_scale": scale,
            "relative_early_late_drift": drift / scale,
            "integrated_autocorrelation_samples": tau_int,
            "effective_time_samples": float(len(clean) / tau_int),
        }
    return record


def _tail_fit_core(items: list[dict], name: str, threshold: float,
                   minimum: int) -> dict:
    """Pool histograms and fit one declared tail interval."""
    edges = counts = None
    for item in items:
        path = item.get("histograms_file")
        if not path:
            continue
        with np.load(path, allow_pickle=False) as data:
            candidate_edges = np.asarray(data[f"{name}_edges"], dtype=float)
            candidate_counts = np.asarray(data[f"{name}_counts"], dtype=float)
        if edges is None:
            edges, counts = candidate_edges, candidate_counts.copy()
        elif not np.array_equal(edges, candidate_edges):
            raise ValueError(f"inconsistent {name} histogram bins")
        else:
            counts += candidate_counts
    base = {"minimum_tail_count": int(minimum), "threshold": float(threshold),
            "fit_ready": False}
    if edges is None:
        return base
    centers = (np.sqrt(edges[:-1] * edges[1:]) if name != "c"
               else 0.5 * (edges[:-1] + edges[1:]))
    widths = np.diff(edges)
    mask = (centers >= threshold) & (counts > 0)
    tail_count = int(np.sum(counts[centers >= threshold]))
    base["tail_count"] = tail_count
    if tail_count < minimum:
        base["reason"] = "tail_count_below_gate"
        return base
    if np.count_nonzero(mask) < 5:
        base["reason"] = "fewer_than_five_occupied_tail_bins"
        return base
    radial_density = counts[mask] / widths[mask]
    jacobian_power = {"c": 2.0, "w": 1.0, "x": 0.0}[name]
    vdf = radial_density / centers[mask] ** jacobian_power
    independent = centers[mask] if name == "c" else np.log(centers[mask])
    dependent = np.log(vdf)
    weights = np.sqrt(counts[mask])
    slope, intercept = np.polyfit(independent, dependent, 1, w=weights)
    fitted = intercept + slope * independent
    weighted_mean = np.average(dependent, weights=weights**2)
    residual = np.sum(weights**2 * (dependent - fitted)**2)
    total = np.sum(weights**2 * (dependent - weighted_mean)**2)
    base.update({
        "fit_ready": True,
        "tail_form": "exp(-gamma*c)" if name == "c" else "power_law",
        "gamma": float(-slope),
        "intercept": float(intercept),
        "weighted_r_squared": float(1.0 - residual / total) if total > 0 else None,
        "occupied_tail_bins": int(np.count_nonzero(mask)),
        "radial_jacobian_power_removed": jacobian_power,
    })
    return base


def pooled_tail_fit(items: list[dict], name: str) -> dict:
    """Fit a tail descriptively and audit independent-replicate uncertainty.

    Particle histograms contain radial samples.  The vector marginal VDFs
    therefore require removal of the radial Jacobian: c**2 for three
    translational dimensions and w for two rotational dimensions.  The c VDF
    is fitted as exp(-gamma*c); w and x are fitted as power laws.  A fit is
    not called asymptotic unless it is stable to raising the threshold and to
    leave-one-realization-out jackknifing.
    """
    minimum = max(int(item["minimum_tail_count"]) for item in items)
    threshold = max(float(item["tail_thresholds"].get(name, np.inf)) for item in items)
    base = _tail_fit_core(items, name, threshold, minimum)
    base["independent_replicates"] = len(items)
    base["asymptotic_claim_ready"] = False
    base["interpretation"] = "intermediate_range_only"
    if not base.get("fit_ready"):
        return base

    jackknife = []
    if len(items) >= 3:
        for omitted in range(len(items)):
            fit = _tail_fit_core(items[:omitted] + items[omitted + 1:], name,
                                 threshold, minimum)
            if fit.get("fit_ready"):
                jackknife.append(float(fit["gamma"]))
    if len(jackknife) == len(items):
        values = np.asarray(jackknife)
        center = float(np.mean(values))
        error = float(np.sqrt((len(values) - 1.0) / len(values)
                              * np.sum((values - center) ** 2)))
        base["gamma_jackknife_standard_error"] = error
        base["gamma_95ci"] = [float(base["gamma"] - 1.96 * error),
                              float(base["gamma"] + 1.96 * error)]

    sensitivity = []
    for factor in (1.0, 1.15, 1.30):
        fit = _tail_fit_core(items, name, threshold * factor, minimum)
        sensitivity.append({"threshold_factor": factor,
                            "threshold": threshold * factor,
                            "fit_ready": bool(fit.get("fit_ready")),
                            "gamma": fit.get("gamma"),
                            "tail_count": fit.get("tail_count", 0)})
    base["threshold_sensitivity"] = sensitivity
    stable_threshold = all(item["fit_ready"] for item in sensitivity)
    if stable_threshold:
        gammas = np.asarray([item["gamma"] for item in sensitivity], dtype=float)
        stable_threshold = (np.ptp(gammas)
                            <= 0.25 * max(abs(float(base["gamma"])), 1.0e-12))
    ci = base.get("gamma_95ci")
    precise = (ci is not None and (ci[1] - ci[0])
               <= 0.5 * max(abs(float(base["gamma"])), 1.0e-12))
    base["asymptotic_claim_ready"] = bool(
        len(items) >= 8 and stable_threshold and precise
        and (base.get("weighted_r_squared") or 0.0) >= 0.95)
    if base["asymptotic_claim_ready"]:
        base["interpretation"] = "asymptotic_candidate"
    return base


def make_figure(cases: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.2), squeeze=False)
    for ax, name in zip(axes.flat, OBSERVABLES):
        arms = sorted({case["arm"] for case in cases})
        for arm in arms:
            subset = [case for case in cases
                      if case["arm"] == arm and case["observables"][name] is not None]
            for ar in sorted({case["aspect_ratio"] for case in subset}):
                values = sorted((case for case in subset
                                 if case["aspect_ratio"] == ar),
                                key=lambda case: case["alpha"])
                if not values:
                    continue
                ax.errorbar([case["alpha"] for case in values],
                            [case["observables"][name]["mean"] for case in values],
                            yerr=[case["observables"][name]["replicate_stderr"]
                                  for case in values], marker="o", ms=3, lw=1,
                            label=f"AR={ar:g}, {arm}")
        ax.set_title(name)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[len(OBSERVABLES):]:
        ax.set_visible(False)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)),
                   frameon=False)
    fig.suptitle("HCS reduced cumulants and orientational correlations")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


# Fixed categorical order (reference palette slots 1-4) plus a marker per
# series so identity never rests on colour alone.
SERIES_STYLE = (("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"),
                ("#eda100", "D"))
CALIBRATED_ALPHA = (0.5, 0.8, 0.95, 1.0)
SWEEP_ROWS = (("theta_rot_over_tr", r"$\theta^H=T_{rot}/T_{tr}$"),
              ("a20", r"$a_{20}^H$"),
              ("a02", r"$a_{02}^H$"), ("a11", r"$a_{11}^H$"),
              ("A_cu", r"$\langle (\mathbf{c}\cdot\hat{\mathbf{u}})^2 - c^2/3\rangle$"))


def ihs_a2(alpha):
    """First Sonine coefficient of smooth inelastic hard spheres in 3D
    (van Noije & Ernst 1998): the reference for a20 without rotation."""
    alpha = np.asarray(alpha, dtype=float)
    return (16.0 * (1.0 - alpha) * (1.0 - 2.0 * alpha**2)
            / (241.0 - 177.0 * alpha + 30.0 * alpha**2 * (1.0 - alpha)))


def _sweep_panel(ax, cases, name, fixed_key, fixed_values, x_key):
    for (color, marker), fixed in zip(SERIES_STYLE, fixed_values):
        subset = sorted((case for case in cases
                         if np.isclose(case[fixed_key], fixed)
                         and case["observables"][name] is not None),
                        key=lambda case: case[x_key])
        if not subset:
            continue
        x = np.array([case[x_key] for case in subset])
        y = np.array([case["observables"][name]["mean"] for case in subset])
        ci = np.array([case["observables"][name]["realization_bootstrap_95ci"]
                       for case in subset], dtype=float)
        label = (rf"AR$={fixed:g}$" if fixed_key == "aspect_ratio"
                 else ("elastic" if fixed >= 1.0 else rf"$\alpha={fixed:g}$"))
        ax.plot(x, y, color=color, lw=2.0, label=label)
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, lw=0)
        # Open markers: alpha interpolated between calibrated artifact nodes.
        calibrated = np.array([any(np.isclose(case["alpha"], CALIBRATED_ALPHA))
                               for case in subset])
        ax.plot(x[calibrated], y[calibrated], marker, color=color, ms=8,
                ls="none", mec="white", mew=1.0)
        ax.plot(x[~calibrated], y[~calibrated], marker, color=color, ms=8,
                ls="none", mfc="white", mew=1.5)
    if not name.startswith("theta_"):
        ax.axhline(0.0, color="0.55", lw=0.8, zorder=0)
    else:
        ax.axhline(1.0, color="0.55", lw=0.8, zorder=0)
    ax.grid(alpha=0.15)


def make_sweep_figure(cases: list[dict], output: Path) -> None:
    """Paper-style figure: rows are observables, columns the two sweeps."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(SWEEP_ROWS), 2, figsize=(9.0, 13.0),
                             sharex="col", squeeze=False)
    for row, (name, label) in enumerate(SWEEP_ROWS):
        _sweep_panel(axes[row, 0], cases, name, "aspect_ratio", (1.5, 2.0, 3.0), "alpha")
        _sweep_panel(axes[row, 1], cases, name, "alpha", (0.5, 0.8, 0.95, 1.0),
                     "aspect_ratio")
        axes[row, 0].set_ylabel(label)
        if name == "a20":
            grid = np.linspace(0.5, 1.0, 101)
            (ihs,) = axes[row, 0].plot(grid, ihs_a2(grid), color="0.35", lw=1.2,
                                       ls="--", label="smooth IHS (Sonine)")
            axes[row, 0].legend(handles=[ihs], frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel(r"$\alpha$")
    axes[-1, 1].set_xlabel("AR")
    axes[0, 0].set_title(r"vs $\alpha$ (open: interpolated $\alpha$)", fontsize=10)
    axes[0, 1].set_title("vs aspect ratio", fontsize=10)
    for column in range(2):
        axes[0, column].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _maxwell_bin_probabilities(edges: np.ndarray, name: str) -> np.ndarray:
    """Exact Maxwellian bin probabilities for dt=3 and dr=2."""
    from scipy.special import erf, kv
    from scipy.integrate import quad
    if name == "c":
        cdf = lambda c: erf(c) - 2.0 * c * np.exp(-c * c) / np.sqrt(np.pi)
        return np.diff(cdf(edges))
    elif name == "w":
        return np.diff(1.0 - np.exp(-edges**2))
    pdf = lambda x: (0.5 * (4.0 * np.pi) * (2.0 * np.pi) * np.pi**-2.5
                     * x**0.25 * kv(0.5, 2.0 * np.sqrt(x)))
    return np.array([quad(pdf, lo, hi)[0]
                     for lo, hi in zip(edges[:-1], edges[1:])])


def _replicate_ratio(items: list[dict], name: str):
    """Marginal/Maxwellian ratio with replicate-bootstrap uncertainty.

    Snapshots within one DSMC realization share particles and are correlated,
    so Poisson bands on pooled particle counts are pseudoreplication.  The
    independent realization is the resampling unit here.
    """
    edges, replicate_counts, totals = None, [], []
    for item in items:
        path = item.get("histograms_file")
        if not path or not Path(path).is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            candidate_edges = np.asarray(data[f"{name}_edges"], dtype=float)
            counts = np.asarray(data[f"{name}_counts"], dtype=float)
            under = float(data[f"{name}_underflow"]) \
                if f"{name}_underflow" in data else 0.0
            over = float(data[f"{name}_overflow"]) \
                if f"{name}_overflow" in data else 0.0
        if edges is None:
            edges = candidate_edges
        elif not np.array_equal(edges, candidate_edges):
            raise ValueError(f"inconsistent {name} histogram bins")
        replicate_counts.append(counts)
        totals.append(float(counts.sum() + under + over))
    if edges is None or not replicate_counts:
        return None
    counts = np.asarray(replicate_counts)
    totals = np.asarray(totals)
    probabilities = _maxwell_bin_probabilities(edges, name)
    pooled_counts = counts.sum(axis=0)
    pooled_expected = totals.sum() * probabilities
    keep = (pooled_counts >= 25) & (pooled_expected > 0.0)
    if not np.any(keep):
        return None
    ratio = pooled_counts[keep] / pooled_expected[keep]
    if len(counts) > 1:
        rng = np.random.default_rng(260922 + {"c": 1, "w": 2, "x": 3}[name])
        draw = rng.integers(0, len(counts), size=(4000, len(counts)))
        boot_counts = counts[draw].sum(axis=1)[:, keep]
        boot_totals = totals[draw].sum(axis=1)
        boot_ratio = boot_counts / (boot_totals[:, None] * probabilities[keep])
        lower, upper = np.quantile(boot_ratio, (0.025, 0.975), axis=0)
    else:
        lower = upper = ratio.copy()
    centers = (0.5 * (edges[:-1] + edges[1:]) if name == "c"
               else np.sqrt(edges[:-1] * edges[1:]))
    return centers[keep], ratio, lower, upper


def _maxwell_density(values: np.ndarray, name: str) -> np.ndarray:
    """Paper Eqs. (5.3a-c), specialized to dt=3 and dr=2."""
    from scipy.special import kv
    values = np.asarray(values, dtype=float)
    if name == "c":
        return np.pi**-1.5 * np.exp(-values**2)
    if name == "w":
        return np.pi**-1.0 * np.exp(-values**2)
    omega_3, omega_2 = 4.0 * np.pi, 2.0 * np.pi
    return (0.5 * omega_3 * omega_2 * np.pi**-2.5
            * values**0.25 * kv(0.5, 2.0 * np.sqrt(values)))


def _representative_inelastic_alphas(by_case: dict) -> tuple[float, ...]:
    """Choose a broad low/mid/near-elastic comparison, not adjacent nodes."""
    available = sorted({key[0] for key in by_case if key[0] < 1.0})
    for preferred in ((0.50, 0.75, 0.95), (0.50, 0.80, 0.95)):
        if all(any(np.isclose(value, target) for value in available)
               for target in preferred):
            return preferred
    if len(available) <= 3:
        return tuple(available)
    indices = np.rint(np.linspace(0, len(available) - 1, 3)).astype(int)
    return tuple(available[index] for index in indices)


def make_marginal_figure(records: list, output: Path) -> None:
    """Paper-Fig.-6 analogue using the actual vector marginal densities."""
    import matplotlib.pyplot as plt

    by_case = defaultdict(list)
    for row, record in records:
        if row["arm"] == "scaled":
            by_case[(float(row["alpha"]), float(row["aspect_ratio"]))].append(record)
    available_ars = sorted({key[1] for key in by_case})
    ars = ((1.5, 2.0, 3.0) if all(value in available_ars
                                  for value in (1.5, 2.0, 3.0))
           else tuple(available_ars[:3]))
    alphas = _representative_inelastic_alphas(by_case)
    labels = {"c": (r"$c$", r"$\phi_c(c)$"),
              "w": (r"$w$", r"$\phi_w(w)$"),
              "x": (r"$x=c^2w^2$", r"$\phi_{cw}(x)$")}
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.5), squeeze=False)
    for row_index, name in enumerate(("c", "w", "x")):
        for column, ar in enumerate(ars):
            ax = axes[row_index, column]
            plotted_x = []
            for (color, marker), alpha in zip(SERIES_STYLE, alphas):
                pooled = _replicate_ratio(by_case.get((alpha, ar), []), name)
                if pooled is None:
                    continue
                x, ratio, lower, upper = pooled
                maxwell = _maxwell_density(x, name)
                density = ratio * maxwell
                ax.plot(x, density, color=color, lw=1.4, marker=marker,
                        markevery=max(1, len(x) // 12), ms=3,
                        label=rf"$\alpha={alpha:g}$")
                ax.fill_between(x, lower * maxwell, upper * maxwell,
                                color=color, alpha=0.16, lw=0)
                plotted_x.append(x)
            if plotted_x:
                positive = np.concatenate(plotted_x)
                positive = positive[positive > 0.0]
                lo, hi = float(np.min(positive)), float(np.max(positive))
                grid = (np.linspace(lo, hi, 400) if name == "c"
                        else np.geomspace(lo, hi, 400))
                ax.plot(grid, _maxwell_density(grid, name), color="0.15",
                        ls="--", lw=1.2, label="Maxwellian")
            ax.set_yscale("log")
            if name != "c":
                ax.set_xscale("log")
            ax.set_xlabel(labels[name][0])
            if column == 0:
                ax.set_ylabel(labels[name][1])
            if row_index == 0:
                ax.set_title(rf"AR$={ar:g}$", fontsize=10)
            ax.grid(alpha=0.15, which="both")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=4,
                   frameon=False)
    fig.suptitle("Stationary HCS marginal distributions (dt=3, dr=2)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_tail_figure(records: list, output: Path) -> None:
    """Thermal-range phi/phi_Maxwell with Sonine predictions.

    The deviations here are percent-level, so the ratio to the Maxwellian is
    shown instead of log densities, where every curve would overlap the
    Gaussian.  Bands resample independent realizations (not correlated
    particle snapshots), and dashed lines are the complete fourth-order
    Sonine marginals from Megias & Santos Eqs. (5.4a-c), specialized to
    three translational and two rotational degrees of freedom.
    """
    import matplotlib.pyplot as plt

    by_case = defaultdict(list)
    for row, record in records:
        if row["arm"] == "scaled":
            by_case[(float(row["alpha"]), float(row["aspect_ratio"]))].append(record)
    available_ars = sorted({key[1] for key in by_case})
    ars = ((1.5, 2.0, 3.0) if all(value in available_ars
                                  for value in (1.5, 2.0, 3.0))
           else tuple(available_ars[:3]))
    alphas = _representative_inelastic_alphas(by_case)
    labels = {"c": (r"$c$", r"$\phi_c/\phi_{c,M}$"),
              "w": (r"$w$", r"$\phi_w/\phi_{w,M}$"),
              "x": (r"$c^2w^2$", r"$\phi_{cw}/\phi_{cw,M}$")}
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.5), sharey="row", squeeze=False)
    for row, name in enumerate(("c", "w", "x")):
        for column, ar in enumerate(ars):
            ax = axes[row, column]
            for (color, marker), alpha in zip(SERIES_STYLE, alphas):
                pooled = _replicate_ratio(by_case.get((alpha, ar), []), name)
                if pooled is None:
                    continue
                x, ratio, lower, upper = pooled
                ax.plot(x, ratio, color=color, lw=1.6, label=rf"$\alpha={alpha:g}$")
                ax.fill_between(x, lower, upper, color=color,
                                alpha=0.18, lw=0)
                # Sonine form with this case's measured cumulants, Eqs. (5.4a-c)
                # for d_t=3, d_r=2: where the data leave it, cumulants beyond
                # fourth order carry the tail.
                items = by_case[(alpha, ar)]
                moments = {
                    key: float(np.mean([item["observables"][key]["mean"]
                                        for item in items]))
                    for key in ("a20", "a02", "a11")
                }
                grid = np.linspace(x.min(), x.max(), 200) if name == "c" \
                    else np.geomspace(x.min(), x.max(), 200)
                if name == "c":
                    sonine = (1.0 + moments["a20"]
                              * (4 * grid**4 - 20 * grid**2 + 15) / 8.0)
                elif name == "w":
                    sonine = (1.0 + moments["a02"]
                              * (4 * grid**4 - 16 * grid**2 + 8) / 8.0)
                else:
                    # Megias & Santos Eq. (5.4c), specialized to dt=3,
                    # dr=2.  Both Bessel orders are 1/2, so their ratio is 1.
                    a20, a02, a11 = (moments["a20"], moments["a02"], moments["a11"])
                    joint = a20 + 2.0 * a11 + a02
                    sonine = (1.0 + 0.5 * joint * grid + 15.0 * a20 / 8.0
                              + 1.5 * a11 + a02
                              - np.sqrt(grid) * (0.5 * (a20 + a02)
                                                 + 1.25 * joint))
                ax.plot(grid, sonine, color=color, lw=1.0, ls="--")
            ax.axhline(1.0, color="0.35", lw=0.9)
            # Truncated Sonine forms turn negative far out (their breakdown);
            # keep the axis on the measured range.
            low, high = ax.get_ylim()
            ax.set_ylim(max(low, 0.3), min(high, 2.0))
            if name != "c":
                ax.set_xscale("log")
                # below these the log bins hold too few counts to read
                ax.set_xlim(left=0.05 if name == "w" else 1.0e-3)
            ax.set_xlabel(labels[name][0])
            if column == 0:
                ax.set_ylabel(labels[name][1])
            if row == 0:
                ax.set_title(rf"AR$={ar:g}$", fontsize=10)
            ax.grid(alpha=0.15)
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
            break
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figure")
    parser.add_argument("--allow-missing", action="store_true",
                        help="summarize the completed tasks and report the rest; "
                             "the campaign verdict is then always false")
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.manifest).open(newline="")))
    records, missing, failed = [], [], []
    for row in rows:
        prefix = Path(row["output_prefix"])
        failed_path = Path(str(prefix) + ".failed.json")
        if failed_path.is_file():
            payload = json.loads(failed_path.read_text())
            failed.append({
                "task_id": int(row["task_id"]),
                "error_type": payload.get("error_type"),
                "error": payload.get("error"),
                "failure_file": str(failed_path),
            })
            continue
        try:
            records.append((row, replicate_record(row)))
        except (FileNotFoundError, RuntimeError):
            missing.append(int(row["task_id"]))
    if (missing or failed) and not args.allow_missing:
        raise SystemExit(
            f"incomplete HCS-NG outputs: missing={missing[:20]}, "
            f"failed={[item['task_id'] for item in failed[:20]]}")
    if not records:
        raise SystemExit("no completed HCS-NG tasks to summarize")
    grouped = defaultdict(list)
    for row, record in records:
        grouped[(float(row["alpha"]), float(row["aspect_ratio"]), row["arm"],
                 float(row.get("initial_theta") or 1.0))].append(record)
    cases = []
    for (alpha, ar, arm, initial_theta), items in sorted(grouped.items()):
        observables = {}
        for name in ANALYSIS_OBSERVABLES:
            values = [item["observables"][name]["mean"] for item in items
                      if item["observables"][name] is not None]
            drifts = [item["observables"][name]["relative_early_late_drift"]
                      for item in items if item["observables"][name] is not None]
            signed = np.array([item["observables"][name]["signed_early_late_drift"]
                               for item in items
                               if item["observables"][name] is not None])
            scales = [item["observables"][name]["drift_scale"]
                      for item in items if item["observables"][name] is not None]
            observables[name] = None if not values else {
                "mean": float(np.mean(values)),
                "replicate_stderr": (float(np.std(values, ddof=1) / np.sqrt(len(values)))
                                     if len(values) > 1 else 0.0),
                "realization_bootstrap_95ci": bootstrap_ci(values),
                "maximum_relative_early_late_drift": float(max(drifts)),
                # Stationarity is judged on the replicate ensemble: a single
                # realization's early/late difference is dominated by Monte
                # Carlo noise whenever the cumulant itself is near zero.
                "replicate_mean_drift": float(np.mean(signed)),
                "replicate_mean_drift_stderr": (
                    float(np.std(signed, ddof=1) / np.sqrt(len(signed)))
                    if len(signed) > 1 else 0.0),
                "drift_tolerance": float(max(
                    0.10 * max(abs(float(np.mean(values))),
                               min(scales)),
                    3.0 * (float(np.std(signed, ddof=1) / np.sqrt(len(signed)))
                           if len(signed) > 1 else 0.0))),
            }
        protocol_v6 = all(
            item["protocol_version"] == "hcs-ng-v6" for item in items)
        correction_fallback_pass = all(
            (((item["result"].get("routing") != "variational_v2")
              or (item["result"].get("correction_fallback_policy") == "base_law"
                  and int(item["result"].get(
                      "evaluation_closure_queries", 0)) > 0
                  and float(item["result"].get(
                      "correction_fallback_fraction_in_evaluation_window", np.inf))
                  < MAXIMUM_EVALUATION_CORRECTION_FALLBACK_FRACTION))
             if protocol_v6 else
             float(item["result"].get("out_of_domain_fraction", 0.0)) < 1.0e-3)
            for item in items)
        closure_runtime_pass = all(
            item["result"].get("negative_energy_repairs", 0) == 0
            and item["result"].get("energy_axis_clamps", 0) == 0
            and item["result"].get("energy_monotonic_repairs", 0) == 0
            for item in items) and correction_fallback_pass
        legacy_engineering_bulk = (
            rows and rows[0]["mode"] == "engineering"
            and all(item["protocol_version"] == "hcs-ng-v3" for item in items))
        bulk_frame_pass = legacy_engineering_bulk or all(
            float(item["result"].get(
                "maximum_bulk_to_thermal_temperature_ratio", np.inf))
            < MAXIMUM_BULK_TO_THERMAL_TEMPERATURE_RATIO for item in items)
        ntc_quality_pass = all(
            item["result"].get("ntc") is not None
            and float(item["result"]["ntc"].get(
                "majorant_violations_per_accepted_pair", np.inf))
            <= MAX_MAJORANT_VIOLATIONS_PER_ACCEPTED_PAIR
            and int(item["result"]["ntc"].get("peak_candidates_per_step", 10**30))
            <= int(item["result"]["ntc"].get("candidate_ceiling_per_step", -1))
            and float(item["result"]["ntc"].get(
                "final_vrmax_over_initial", np.inf)) <= 3.0
            and float(item["result"]["ntc"].get(
                "repeated_particle_pair_fraction", np.inf)) <= 1.0e-2
            for item in items)
        runtime_pass = closure_runtime_pass and ntc_quality_pass and bulk_frame_pass
        performance_pass = all(
            item["result"].get("closure_overhead_fraction", 0.0) < 0.15
            for item in items)
        stationarity = {}
        for name in STATIONARITY_OBSERVABLES:
            curves = [item["series"][name] for item in items
                      if name in item["series"]]
            finite_curves = [curve[np.isfinite(curve)] for curve in curves]
            if not finite_curves or min(map(len, finite_curves)) == 0:
                stationarity[name] = {
                    "pass": False, "reason": "observable_missing",
                    "absolute_tolerance": STATIONARITY_ABSOLUTE_TOLERANCE[name],
                    "n_samples": 0,
                }
                continue
            stationarity[name] = replicate_stationarity(finite_curves, name)
        stationarity_pass = all(
            stationarity[name]["pass"] for name in STATIONARITY_OBSERVABLES)
        achieved_horizons = [
            (float(item["result"].get("cpp", 0.0)) if alpha >= 1.0
             else (1.0 - alpha**2) * float(item["result"].get("cpp", 0.0)))
            for item in items]
        required_horizon = max(
            item["required_dissipation_horizon"] for item in items)
        horizon_pass = all(value + 1.0e-8 >= required_horizon
                           for value in achieved_horizons)
        theta_margin_values = [item["result"].get(
            "minimum_theta_hull_log_margin") for item in items]
        finite_theta_margins = [float(value) for value in theta_margin_values
                                if value is not None and np.isfinite(value)]
        legacy_engineering_margin = (
            rows and rows[0]["mode"] == "engineering"
            and all(item["protocol_version"] == "hcs-ng-v3" for item in items))
        theta_domain_margin_pass = (
            alpha >= 1.0 or legacy_engineering_margin
            or (len(finite_theta_margins) == len(items)
                and min(finite_theta_margins)
                >= MINIMUM_THETA_HULL_LOG_MARGIN))
        runtime_pass = runtime_pass and theta_domain_margin_pass
        tails = {name: sum(int(item["tail_counts"].get(name, 0)) for item in items)
                 for name in ("c", "w", "x")}
        tail_fits = {name: pooled_tail_fit(items, name)
                     for name in ("c", "w", "x")}
        cases.append({
            "alpha": alpha, "aspect_ratio": ar, "arm": arm,
            "initial_theta": initial_theta,
            "n_replicates": len(items),
            "sampling_complete": all(item["sampling_complete"] for item in items),
            "required_dissipation_horizon": required_horizon,
            "minimum_achieved_dissipation_horizon": min(achieved_horizons),
            "dissipation_horizon_pass": horizon_pass,
            "minimum_theta_hull_log_margin": (
                min(finite_theta_margins) if finite_theta_margins else None),
            "theta_domain_margin_pass": theta_domain_margin_pass,
            "runtime_pass": runtime_pass,
            "closure_runtime_pass": closure_runtime_pass,
            "correction_fallback_pass": correction_fallback_pass,
            "maximum_correction_fallback_fraction": max(
                float(item["result"].get(
                    "correction_fallback_fraction",
                    item["result"].get("out_of_domain_fraction", 0.0)))
                for item in items),
            "maximum_correction_fallback_fraction_in_evaluation_window": max(
                float(item["result"].get(
                    "correction_fallback_fraction_in_evaluation_window",
                    item["result"].get("out_of_domain_fraction", 0.0)))
                for item in items),
            "bulk_frame_pass": bulk_frame_pass,
            "maximum_bulk_to_thermal_temperature_ratio": max(
                float(item["result"].get(
                    "maximum_bulk_to_thermal_temperature_ratio", 0.0))
                for item in items),
            "ntc_quality_pass": ntc_quality_pass,
            "maximum_ntc_majorant_violation_fraction": max(
                float((item["result"].get("ntc") or {}).get(
                    "majorant_violation_fraction", np.inf)) for item in items),
            "maximum_ntc_majorant_violations_per_accepted_pair": max(
                float((item["result"].get("ntc") or {}).get(
                    "majorant_violations_per_accepted_pair", np.inf))
                for item in items),
            "maximum_final_vrmax_over_initial": max(
                float((item["result"].get("ntc") or {}).get(
                    "final_vrmax_over_initial", np.inf)) for item in items),
            "maximum_ntc_candidates_per_step": max(
                int((item["result"].get("ntc") or {}).get(
                    "peak_candidates_per_step", -1)) for item in items),
            "maximum_repeated_particle_pair_fraction": max(
                float((item["result"].get("ntc") or {}).get(
                    "repeated_particle_pair_fraction", np.inf))
                for item in items),
            "maximum_peak_rss_mib": max(
                float(item["result"].get("peak_rss_mib", np.inf)) for item in items),
            "performance_pass": performance_pass,
            "maximum_closure_overhead_fraction": max(
                float(item["result"].get("closure_overhead_fraction", 0.0))
                for item in items),
            "stationarity_pass": stationarity_pass,
            "stationarity": stationarity,
            "observables": observables, "pooled_tail_counts": tails,
            "tail_fit_ready": {name: fit["fit_ready"]
                               for name, fit in tail_fits.items()},
            "tail_fits": tail_fits,
        })

    equivalence = []
    by_physical = defaultdict(dict)
    campaign_mode = rows[0]["mode"] if rows else None
    for (alpha, ar, arm, initial_theta), items in grouped.items():
        physical = ((alpha, ar) if campaign_mode in ("stability-sentinel", "stability")
                    else (alpha, ar, initial_theta))
        by_physical[physical].setdefault(arm, {}).update({
            (initial_theta, item["replicate"], item["seed"]): item
            for item in items})
    for physical, arms in sorted(by_physical.items()):
        for control in ("unscaled", "dt_half"):
            if "scaled" not in arms or control not in arms:
                continue
            paired_keys = sorted(set(arms["scaled"]) & set(arms[control]))
            comparisons, passed = {}, True
            for name in CONTROL_OBSERVABLES:
                differences = []
                for key in paired_keys:
                    left = arms["scaled"][key]["observables"][name]
                    right = arms[control][key]["observables"][name]
                    if left is not None and right is not None:
                        differences.append(left["mean"] - right["mean"])
                if not differences:
                    continue
                differences = np.asarray(differences, dtype=float)
                mean_difference = float(np.mean(differences))
                paired_stderr = (float(np.std(differences, ddof=1)
                                       / np.sqrt(len(differences)))
                                 if len(differences) > 1 else 0.0)
                error = abs(mean_difference)
                is_temperature_ratio = name == LOG_THETA
                floor = 0.01 if is_temperature_ratio else 0.002
                statistical_resolution = 3.0 * paired_stderr
                # A large standard error must not make the control easier to
                # pass.  First require consistency with zero at three SE, but
                # also cap that statistical resolution.  Formal equivalence
                # at the tighter practical floor is reported separately; it
                # is not manufactured from a non-significant difference.
                resolution_limit = 0.02 if is_temperature_ratio else 0.01
                tolerance = max(floor, statistical_resolution)
                consistency_pass = error <= tolerance
                precision_pass = (len(differences) >= 4
                                  and statistical_resolution <= resolution_limit)
                practical_equivalence_95 = (
                    error + 1.96 * paired_stderr <= floor)
                comparisons[name] = {"paired_mean_difference": mean_difference,
                                     "absolute_difference": error,
                                     "paired_stderr": paired_stderr,
                                     "n_pairs": len(differences),
                                     "practical_floor": floor,
                                     "statistical_resolution_3se": statistical_resolution,
                                     "resolution_limit": resolution_limit,
                                     "tolerance": tolerance,
                                     "statistical_consistency_pass": bool(
                                         consistency_pass),
                                     "precision_pass": bool(precision_pass),
                                     "practical_equivalence_95": bool(
                                         practical_equivalence_95),
                                     "pass": bool(consistency_pass
                                                  and precision_pass)}
                passed &= consistency_pass and precision_pass
            equivalence.append({"alpha": physical[0], "aspect_ratio": physical[1],
                                "initial_theta": (
                                    None if len(physical) == 2 else physical[2]),
                                "control_arm": control,
                                "comparison_kind": (
                                    "paired_consistency_with_precision_gate"),
                                "paired_realizations": len(paired_keys),
                                "pass": bool(passed), "observables": comparisons})
    attraction = []
    if rows and rows[0]["mode"] in ("stability-sentinel", "stability"):
        by_attractor = defaultdict(list)
        for case in cases:
            if case["arm"] == "scaled":
                by_attractor[(case["alpha"], case["aspect_ratio"])].append(case)
        for (alpha, ar), starts in sorted(by_attractor.items()):
            starts.sort(key=lambda item: item["initial_theta"])
            entry = {"alpha": alpha, "aspect_ratio": ar,
                     "initial_thetas": [item["initial_theta"] for item in starts],
                     "pass": False}
            if len(starts) != 2 or any(
                    item["observables"].get(LOG_THETA) is None for item in starts):
                entry["reason"] = "two_complete_initial_conditions_required"
            else:
                low, high = (item["observables"][LOG_THETA] for item in starts)
                difference = abs(float(high["mean"] - low["mean"]))
                stderr = float(np.hypot(low["replicate_stderr"],
                                        high["replicate_stderr"]))
                resolution = 3.0 * stderr
                tolerance = max(0.03, resolution)
                passed = difference <= tolerance and resolution <= 0.05
                entry.update({
                    "absolute_log_theta_difference": difference,
                    "combined_stderr": stderr,
                    "statistical_resolution_3se": resolution,
                    "tolerance": tolerance,
                    "pass": bool(passed),
                    "reason": None if passed else "late_attractors_do_not_agree",
                })
            attraction.append(entry)

    # A fitted asymptotic form is not scientifically discriminating if the
    # exact elastic Gaussian control passes the same candidate gate.
    elastic_negative_control_pass = {}
    for name in ("c", "w", "x"):
        elastic_fits = [case["tail_fits"][name] for case in cases
                        if case["alpha"] >= 1.0 and case["arm"] == "scaled"]
        elastic_negative_control_pass[name] = bool(
            elastic_fits and all(not fit.get("asymptotic_claim_ready", False)
                                 for fit in elastic_fits))
    for case in cases:
        for name, fit in case["tail_fits"].items():
            fit["elastic_negative_control_pass"] = elastic_negative_control_pass[name]
            fit["scientific_claim_ready"] = bool(
                fit.get("asymptotic_claim_ready", False)
                and elastic_negative_control_pass[name]
                and case["stationarity_pass"]
                and case["runtime_pass"]
                and case["dissipation_horizon_pass"])

    mode = rows[0]["mode"] if rows else None
    arm_dt = {
        arm: sorted({float(row["dt"]) for row in rows if row["arm"] == arm})
        for arm in sorted({row["arm"] for row in rows})
    }
    protocols = sorted({row.get("protocol_version", "hcs-ng-v1") for row in rows})
    variants = sorted({row.get("model_variant", "baseline") for row in rows})
    correction_flags = sorted({row.get("invariant_corrections", "false")
                               for row in rows})
    control_coverage_pass = True
    if mode == "engineering":
        physical_cases = {(float(row["alpha"]), float(row["aspect_ratio"]))
                          for row in rows}
        for control in ("unscaled", "dt_half"):
            covered = {(item["alpha"], item["aspect_ratio"])
                       for item in equivalence
                       if item["control_arm"] == control and item["pass"]}
            control_coverage_pass &= covered == physical_cases
    elif mode == "stability-sentinel":
        dt_cases = {(0.50, 1.35), (0.50, 3.0), (0.80, 2.0),
                    (0.95, 2.0), (1.00, 3.0)}
        passed_dt = {(item["alpha"], item["aspect_ratio"])
                     for item in equivalence
                     if item["control_arm"] == "dt_half" and item["pass"]}
        control_coverage_pass = passed_dt == dt_cases
    # Engineering is a short numerical-control pilot. Its stationarity
    # diagnostics remain visible, but only the long-time sentinel/stability
    # stages are authorized to certify an HCS attractor.
    stationarity_required = mode != "engineering"
    physics_verdict = (bool(cases) and not missing and not failed
                       and all(case["sampling_complete"] and case["runtime_pass"]
                               and (case["stationarity_pass"]
                                    or not stationarity_required)
                               and case["dissipation_horizon_pass"]
                               for case in cases)
                       and all(item["pass"] for item in equivalence)
                       and control_coverage_pass
                       and (mode not in ("stability-sentinel", "stability")
                            or (attraction and all(item["pass"]
                                                   for item in attraction))))
    performance_verdict = bool(cases) and all(
        case["performance_pass"] for case in cases)
    study_verdict = physics_verdict and performance_verdict
    # The angular artifact is deliberately evidence-only.  A successful HCS
    # study validates this campaign, not general cross-flow deployment.
    deployment_eligible = variants == ["baseline"]
    production_verdict = study_verdict and deployment_eligible
    artifact_hashes = sorted({value for _, item in records
                              if (value := item["result"].get("artifact_sha256"))})
    summary = {"protocol_version": protocols[0] if len(protocols) == 1 else None,
               "mode": mode,
               "arm_dt": arm_dt,
               "model_variant": variants[0] if len(variants) == 1 else None,
               "invariant_corrections": (
                   correction_flags[0] == "true" if len(correction_flags) == 1 else None),
               "n_tasks": len(rows), "n_cases": len(cases),
               "n_completed_tasks": len(records), "missing_tasks": missing,
               "failed_tasks": failed,
               "artifact_sha256": artifact_hashes[0] if len(artifact_hashes) == 1 else None,
               "physics_campaign_pass": physics_verdict,
               "stationarity_required_for_verdict": stationarity_required,
               "long_time_stability_campaign_pass": bool(
                   mode in ("stability-sentinel", "stability") and study_verdict),
               "engineering_control_coverage_pass": control_coverage_pass,
               "performance_campaign_pass": performance_verdict,
               "study_campaign_pass": study_verdict,
               "artifact_deployment_eligible": deployment_eligible,
               "production_campaign_pass": production_verdict,
               "campaign_pass": study_verdict,
               "scientific_outputs_released": bool(
                   study_verdict and mode in ("sweep", "map", "tails")),
               "elastic_tail_negative_control_pass": elastic_negative_control_pass,
               "cases": cases, "two_sided_attraction": attraction,
               "scaled_unscaled_equivalence": equivalence}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.figure:
        figure = Path(args.figure)
        if mode in ("sweep", "tails") and study_verdict:
            if mode == "sweep":
                scaled = [case for case in cases if case["arm"] == "scaled"]
                make_sweep_figure(scaled, figure)
            make_marginal_figure(
                records, (figure.with_name(figure.stem + "_marginals.png")
                          if mode == "sweep" else figure))
            make_tail_figure(
                records, figure.with_name(figure.stem + "_sonine_ratios.png"))
        else:
            make_figure(cases, figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
