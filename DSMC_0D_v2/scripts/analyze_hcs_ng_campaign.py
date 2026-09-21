#!/usr/bin/env python3
"""Aggregate HCS non-Gaussian replicates with time-correlation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


OBSERVABLES = ("theta", "a20", "a02", "a11", "A_cu", "A_cw_quadrupolar")


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
    result = {}
    for name in OBSERVABLES:
        result[name] = np.array([
            float(row[name]) if row.get(name, "") not in ("", None) else np.nan
            for row in rows], dtype=float)
    return result


def replicate_record(row: dict[str, str]) -> dict:
    prefix = Path(row["output_prefix"])
    result_path = Path(str(prefix) + ".json")
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text())
    summary = result.get("non_gaussian") or {}
    series = load_series(Path(str(prefix) + "_ng_moments.csv"))
    record = {
        "task_id": int(row["task_id"]), "result": result,
        "sampling_complete": bool(summary.get("sampling_complete", False)),
        "tail_counts": summary.get("tail_counts", {}),
        "tail_thresholds": summary.get("tail_thresholds", {}),
        "histograms_file": summary.get("histograms_file"),
        "minimum_tail_count": int(summary.get("minimum_tail_count", 1000)),
        "observables": {},
    }
    for name, values in series.items():
        clean = values[np.isfinite(values)]
        if not len(clean):
            record["observables"][name] = None; continue
        third = max(1, len(clean) // 3)
        drift = abs(float(np.mean(clean[-third:]) - np.mean(clean[:third])))
        scale = max(abs(float(np.mean(clean))), 0.05 if name != "theta" else 1e-12)
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


def pooled_tail_fit(items: list[dict], name: str) -> dict:
    """Fit the declared asymptotic form only after the tail-count gate.

    Particle histograms contain radial samples.  The vector marginal VDFs
    therefore require removal of the radial Jacobian: c**2 for three
    translational dimensions and w for two rotational dimensions.  The c VDF
    is fitted as exp(-gamma*c); w and x are fitted as power laws.
    """
    edges = counts = None
    for item in items:
        path = item.get("histograms_file")
        if not path:
            continue
        with np.load(path, allow_pickle=False) as data:
            candidate_edges = np.asarray(data[f"{name}_edges"], dtype=float)
            candidate_counts = np.asarray(data[f"{name}_counts"], dtype=float)
        if edges is None:
            edges = candidate_edges
            counts = candidate_counts
        elif not np.array_equal(edges, candidate_edges):
            raise ValueError(f"inconsistent {name} histogram bins")
        else:
            counts += candidate_counts
    tail_count = sum(int(item["tail_counts"].get(name, 0)) for item in items)
    minimum = max(int(item["minimum_tail_count"]) for item in items)
    threshold = max(float(item["tail_thresholds"].get(name, np.inf)) for item in items)
    base = {"tail_count": tail_count, "minimum_tail_count": minimum,
            "threshold": threshold, "fit_ready": False}
    if edges is None or tail_count < minimum:
        return base
    centers = np.sqrt(edges[:-1] * edges[1:]) if name != "c" \
        else 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    mask = (centers >= threshold) & (counts > 0)
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


def make_figure(cases: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), squeeze=False)
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
SWEEP_ROWS = (("theta", r"$\theta^H$"), ("a20", r"$a_{20}^H$"),
              ("a02", r"$a_{02}^H$"), ("a11", r"$a_{11}^H$"))


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
    if name != "theta":
        ax.axhline(0.0, color="0.55", lw=0.8, zorder=0)
    else:
        ax.axhline(1.0, color="0.55", lw=0.8, zorder=0)
    ax.grid(alpha=0.15)


def make_sweep_figure(cases: list[dict], output: Path) -> None:
    """Paper-style figure: rows are observables, columns the two sweeps."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(9.0, 11.0), sharex="col", squeeze=False)
    for row, (name, label) in enumerate(SWEEP_ROWS):
        _sweep_panel(axes[row, 0], cases, name, "aspect_ratio", (1.5, 2.0, 3.0), "alpha")
        _sweep_panel(axes[row, 1], cases, name, "alpha", (0.5, 0.8, 0.95, 1.0),
                     "aspect_ratio")
        axes[row, 0].set_ylabel(label)
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


def _pooled_density(items: list[dict], name: str):
    edges, counts = None, None
    for item in items:
        path = item.get("histograms_file")
        if not path or not Path(path).is_file():
            continue
        with np.load(path) as data:
            edges = data[f"{name}_edges"]
            counts = data[f"{name}_counts"] if counts is None else counts + data[f"{name}_counts"]
    if counts is None or counts.sum() == 0:
        return None
    width = np.diff(edges)
    centers = (0.5 * (edges[:-1] + edges[1:]) if name == "c"
               else np.sqrt(edges[:-1] * edges[1:]))
    density = counts / (counts.sum() * width)
    # Remove the radial Jacobian so the curves are phi_c(c), phi_w(w), and the
    # distribution of x=c^2w^2, matching Megias & Santos Eqs. (5.1).
    if name == "c":
        density = density / (4.0 * np.pi * centers**2)
    elif name == "w":
        density = density / (2.0 * np.pi * centers)
    # Bins with a handful of counts carry ~30% Poisson noise and read as
    # spurious structure on a log axis; show only resolved bins.
    keep = counts >= 20
    return centers[keep], density[keep]


def make_tail_figure(records: list, output: Path) -> None:
    """Marginal distributions on a 3x3 alpha x AR grid (cf. paper Fig. 6)."""
    import matplotlib.pyplot as plt
    from scipy.special import kv

    by_case = defaultdict(list)
    for row, record in records:
        by_case[(float(row["alpha"]), float(row["aspect_ratio"]))].append(record)
    ars, alphas = (1.5, 2.0, 3.0), (0.5, 0.8, 0.95)
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.0), squeeze=False)
    maxwell = {
        "c": lambda c: np.pi**-1.5 * np.exp(-c**2),
        "w": lambda w: np.exp(-w**2) / np.pi,
        # Eq. (5.3c) with d_t=3, d_r=2.
        "x": lambda x: 0.5 * (4.0 * np.pi) * (2.0 * np.pi) * np.pi**-2.5
        * x**0.25 * kv(0.5, 2.0 * np.sqrt(x)),
    }
    labels = {"c": (r"$c$", r"$\phi_c$"), "w": (r"$w$", r"$\phi_w$"),
              "x": (r"$c^2w^2$", r"$\phi_{cw}$")}
    for row, name in enumerate(("c", "w", "x")):
        for column, ar in enumerate(ars):
            ax = axes[row, column]
            grid = None
            for (color, marker), alpha in zip(SERIES_STYLE, alphas):
                pooled = _pooled_density(by_case.get((alpha, ar), []), name)
                if pooled is None:
                    continue
                x, y = pooled
                grid = x if grid is None else grid
                ax.plot(x, y, marker, color=color, ms=3, ls="none",
                        label=rf"$\alpha={alpha:g}$")
            if grid is not None:
                ref = np.linspace(grid.min(), grid.max(), 400) if name == "c" \
                    else np.geomspace(grid.min(), grid.max(), 400)
                ax.plot(ref, maxwell[name](ref), color="0.2", lw=1.2, label="Maxwellian")
            ax.set_yscale("log")
            if name != "c":
                ax.set_xscale("log")
            ax.set_xlabel(labels[name][0]); ax.set_ylabel(labels[name][1])
            if row == 0:
                ax.set_title(rf"AR$={ar:g}$", fontsize=10)
            ax.grid(alpha=0.15)
    for ax in axes[0]:
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
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.manifest).open(newline="")))
    records, missing = [], []
    for row in rows:
        try:
            records.append((row, replicate_record(row)))
        except FileNotFoundError:
            missing.append(int(row["task_id"]))
    if missing:
        raise SystemExit(f"missing HCS-NG outputs for tasks {missing[:20]}")
    grouped = defaultdict(list)
    for row, record in records:
        grouped[(float(row["alpha"]), float(row["aspect_ratio"]), row["arm"])].append(record)
    cases = []
    for (alpha, ar, arm), items in sorted(grouped.items()):
        observables = {}
        for name in OBSERVABLES:
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
        runtime_pass = all(
            item["result"].get("negative_energy_repairs", 0) == 0
            and item["result"].get("energy_axis_clamps", 0) == 0
            and item["result"].get("energy_monotonic_repairs", 0) == 0
            and item["result"].get("out_of_domain_fraction", 0.0) < 1.0e-3
            for item in items)
        performance_pass = all(
            item["result"].get("closure_overhead_fraction", 0.0) < 0.15
            for item in items)
        stationarity_pass = all(
            value is None
            or abs(value["replicate_mean_drift"]) <= value["drift_tolerance"]
            for value in observables.values())
        tails = {name: sum(int(item["tail_counts"].get(name, 0)) for item in items)
                 for name in ("c", "w", "x")}
        tail_fits = {name: pooled_tail_fit(items, name)
                     for name in ("c", "w", "x")}
        cases.append({
            "alpha": alpha, "aspect_ratio": ar, "arm": arm,
            "n_replicates": len(items),
            "sampling_complete": all(item["sampling_complete"] for item in items),
            "runtime_pass": runtime_pass, "performance_pass": performance_pass,
            "maximum_closure_overhead_fraction": max(
                float(item["result"].get("closure_overhead_fraction", 0.0))
                for item in items),
            "stationarity_pass": stationarity_pass,
            "observables": observables, "pooled_tail_counts": tails,
            "tail_fit_ready": {name: fit["fit_ready"]
                               for name, fit in tail_fits.items()},
            "tail_fits": tail_fits,
        })

    equivalence = []
    by_physical = defaultdict(dict)
    for case in cases:
        by_physical[(case["alpha"], case["aspect_ratio"])][case["arm"]] = case
    for physical, arms in sorted(by_physical.items()):
        if "scaled" not in arms or "unscaled" not in arms:
            continue
        comparisons, passed = {}, True
        for name in OBSERVABLES:
            left, right = arms["scaled"]["observables"][name], arms["unscaled"]["observables"][name]
            if left is None or right is None:
                continue
            error = abs(left["mean"] - right["mean"])
            tolerance = max(0.02, 3.0 * np.hypot(left["replicate_stderr"],
                                                 right["replicate_stderr"]))
            comparisons[name] = {"absolute_difference": error, "tolerance": tolerance,
                                 "pass": bool(error <= tolerance)}
            passed &= error <= tolerance
        equivalence.append({"alpha": physical[0], "aspect_ratio": physical[1],
                            "pass": bool(passed), "observables": comparisons})
    mode = rows[0]["mode"] if rows else None
    physics_verdict = (bool(cases)
                       and all(case["sampling_complete"] and case["runtime_pass"]
                               and case["stationarity_pass"] for case in cases)
                       and all(item["pass"] for item in equivalence))
    production_verdict = physics_verdict and all(
        case["performance_pass"] for case in cases)
    artifact_hashes = sorted({item[1]["result"].get("artifact_sha256")
                              for item in records})
    summary = {"mode": mode, "n_tasks": len(rows), "n_cases": len(cases),
               "artifact_sha256": artifact_hashes[0] if len(artifact_hashes) == 1 else None,
               "physics_campaign_pass": physics_verdict,
               "production_campaign_pass": production_verdict,
               "campaign_pass": production_verdict, "cases": cases,
               "scaled_unscaled_equivalence": equivalence}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.figure:
        figure = Path(args.figure)
        if mode == "sweep":
            scaled = [case for case in cases if case["arm"] == "scaled"]
            make_sweep_figure(scaled, figure)
            make_tail_figure(records, figure.with_name(figure.stem + "_tails.png"))
        else:
            make_figure(cases, figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
