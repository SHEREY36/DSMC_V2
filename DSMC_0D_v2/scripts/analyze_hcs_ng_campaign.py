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
            observables[name] = None if not values else {
                "mean": float(np.mean(values)),
                "replicate_stderr": (float(np.std(values, ddof=1) / np.sqrt(len(values)))
                                     if len(values) > 1 else 0.0),
                "realization_bootstrap_95ci": bootstrap_ci(values),
                "maximum_relative_early_late_drift": float(max(drifts)),
            }
        runtime_pass = all(
            item["result"].get("negative_energy_repairs", 0) == 0
            and item["result"].get("energy_axis_clamps", 0) == 0
            and item["result"].get("energy_monotonic_repairs", 0) == 0
            and item["result"].get("out_of_domain_fraction", 0.0) < 1.0e-3
            for item in items)
        performance_pass = all(
            item["result"].get("closure_overhead_fraction", 0.0) < 0.05
            for item in items)
        stationarity_pass = all(
            value is None or value["maximum_relative_early_late_drift"] <= 0.10
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
                                 "pass": error <= tolerance}
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
        make_figure(cases, Path(args.figure))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
