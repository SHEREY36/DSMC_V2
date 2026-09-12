#!/usr/bin/env python3
"""Plot HCS cooling and theta attraction, and write quantitative gate results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def relative_linear_drift(x: np.ndarray, y: np.ndarray) -> float:
    """Return the fitted change across a window relative to its mean."""
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
    span = float(x[-1] - x[0]) if len(x) > 1 else 0.0
    return abs(slope) * span / max(abs(float(np.mean(y))), 1.0e-12)


def replicate_mean_relative_drift(replicates: list[np.ndarray]) -> float:
    """Measure late drift after averaging independent DSMC replicates.

    DSMC trajectories contain finite-particle collision noise.  The campaign
    deliberately runs three independent replicates, so the case-level gate
    must judge their mean trajectory rather than fail on one noisy slope.
    Individual drifts remain in the JSON as diagnostics.
    """
    length = min(len(values) for values in replicates)
    tau = replicates[0][:length, 0]
    theta_mean = np.mean(
        np.asarray([values[:length, 1] for values in replicates]), axis=0)
    start = max(1, int(0.7 * length))
    return relative_linear_drift(tau[start:], theta_mean[start:])


def analyse(row: dict[str, str]) -> tuple[dict, np.ndarray]:
    prefix = row["output_prefix"]
    data = np.atleast_2d(np.loadtxt(Path(prefix + ".txt")))
    diagnostics = json.loads(Path(prefix + ".json").read_text())
    tau, ttr, trot, total = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    theta = ttr / trot
    start = max(1, int(0.7 * len(theta)))
    x, y = tau[start:], theta[start:]
    mean = float(np.mean(y))
    drift = relative_linear_drift(x, y)
    target = None if not row["target_theta"] else float(row["target_theta"])
    alpha = float(row["alpha"])
    total_change = float(total[-1] / total[0] - 1.0)
    # Inelastic HCS cools forever; elastic HCS conserves total energy.  Asking
    # alpha=1 to cool would reject the exact physical limit for the wrong
    # reason.  Two percent is far above roundoff yet tight enough to catch a
    # genuinely dissipative elastic implementation.
    energy_behavior_pass = (
        abs(total_change) <= 0.02 if np.isclose(alpha, 1.0)
        else total_change < 0.0)
    result = {
        "task_id": int(row["task_id"]), "tier": row["tier"],
        "alpha": alpha, "aspect_ratio": float(row["aspect_ratio"]),
        "theta0": float(row["theta0"]), "target_theta": target,
        "replicate": int(row.get("replicate", 0)),
        "late_theta_mean": mean,
        "late_theta_std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        "late_relative_drift": drift,
        "relative_target_error": None if target is None else abs(mean - target) / target,
        "relative_total_energy_change": total_change,
        "energy_behavior_pass": bool(energy_behavior_pass and np.all(total > 0.0)),
        "bounded": bool(np.all(np.isfinite(theta)) and np.all(theta > 0.0)
                        and np.max(theta) < 10.0),
        "negative_energy_repairs": int(diagnostics["negative_energy_repairs"]),
        "energy_axis_clamps": int(diagnostics.get("energy_axis_clamps", 0)),
        "energy_monotonic_repairs": int(
            diagnostics.get("energy_monotonic_repairs", 0)),
        "maximum_energy_monotonic_repair": float(
            diagnostics.get("maximum_energy_monotonic_repair", 0.0)),
        "out_of_domain_fraction": float(diagnostics["out_of_domain_fraction"]),
        "out_of_domain_fraction_by_feature": diagnostics.get(
            "out_of_domain_fraction_by_feature", {}),
        "sampling_excursion_fraction_by_feature": diagnostics.get(
            "sampling_excursion_fraction_by_feature", {}),
        "closure_overhead_fraction": float(diagnostics["closure_overhead_fraction"]),
        "artifact": diagnostics.get("artifact"),
        "artifact_sha256": diagnostics.get("artifact_sha256"),
        "performance_gate_pass": float(diagnostics["closure_overhead_fraction"]) < 0.05,
    }
    return result, np.column_stack((tau, theta, total / total[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/hcs_validation.csv")
    parser.add_argument("--figure", default="reports/figures/hcs_validation.png")
    parser.add_argument("--summary", default="results/hcs_validation/summary.json")
    args = parser.parse_args()

    records, series, missing = [], [], []
    manifest_rows = load_manifest(Path(args.manifest))
    for row in manifest_rows:
        trajectory_path = Path(row["output_prefix"] + ".txt")
        diagnostics_path = Path(row["output_prefix"] + ".json")
        if trajectory_path.exists() and diagnostics_path.exists():
            record, values = analyse(row)
            records.append(record)
            series.append((row, values))
        else:
            missing.append(int(row["task_id"]))
    if missing:
        raise SystemExit(f"missing HCS outputs for tasks {missing}")

    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["alpha"], record["aspect_ratio"])].append(record)
    cases = []
    for (alpha, ar), items in sorted(grouped.items()):
        means = np.array([item["late_theta_mean"] for item in items])
        target = items[0]["target_theta"]
        by_start = defaultdict(list)
        for item in items:
            by_start[item["theta0"]].append(item["late_theta_mean"])
        start_means = np.array([np.mean(by_start[start]) for start in sorted(by_start)])
        convergence = ((float(np.ptp(start_means))
                        / max(float(np.mean(start_means)), 1.0e-12))
                       if len(start_means) > 1 else None)
        replicate_cv = max(
            (float(np.std(values, ddof=1) / max(abs(np.mean(values)), 1.0e-12))
             if len(values) > 1 else 0.0)
            for values in by_start.values())
        matching_series = [(row, values) for row, values in series
                           if float(row["alpha"]) == alpha
                           and float(row["aspect_ratio"]) == ar]
        mean_drift_by_start = {}
        for theta0 in sorted(by_start):
            replicate_series = [values for row, values in matching_series
                                if float(row["theta0"]) == theta0]
            mean_drift_by_start[f"{theta0:g}"] = replicate_mean_relative_drift(
                replicate_series)
        maximum_mean_drift = max(mean_drift_by_start.values())
        maximum_individual_drift = max(
            item["late_relative_drift"] for item in items)
        overhead = np.array([item["closure_overhead_fraction"] for item in items])
        physics_pass = (len(start_means) > 1 and convergence <= 0.10
                        and replicate_cv <= 0.10
                        and all(item["bounded"] and item["energy_behavior_pass"]
                                and item["negative_energy_repairs"] == 0
                                and item["energy_axis_clamps"] == 0
                                and item["energy_monotonic_repairs"] == 0
                                and item["out_of_domain_fraction"] < 1.0e-3
                                for item in items)
                        and maximum_mean_drift <= 0.10
                        and (target is None
                             or abs(float(np.mean(means)) - target) / target <= 0.10))
        production_pass = physics_pass and all(
            item["performance_gate_pass"] for item in items)
        cases.append({"alpha": alpha, "aspect_ratio": ar, "tier": items[0]["tier"],
                      "target_theta": target, "mean_theta": float(np.mean(means)),
                      "initial_condition_spread": convergence,
                      "replicate_coefficient_of_variation": replicate_cv,
                      "replicate_mean_relative_drift_by_theta0": mean_drift_by_start,
                      "maximum_replicate_mean_relative_drift": maximum_mean_drift,
                      "maximum_individual_relative_drift": maximum_individual_drift,
                      "mean_closure_overhead_fraction": float(np.mean(overhead)),
                      "maximum_closure_overhead_fraction": float(np.max(overhead)),
                      "n_performance_failures": int(np.count_nonzero(overhead >= 0.05)),
                      "physics_pass": bool(physics_pass),
                      "production_pass": bool(production_pass)})

    overhead = np.array([record["closure_overhead_fraction"] for record in records])
    artifact_digests = {record["artifact_sha256"] for record in records
                        if record.get("artifact_sha256")}
    artifact_consistent = bool(len(artifact_digests) == 1
                               and all(record.get("artifact_sha256")
                                       for record in records))
    summary = {"criteria": {"target_relative_error_max": 0.10,
                             "replicate_mean_relative_drift_max": 0.10,
                             "initial_condition_spread_max": 0.10,
                             "replicate_coefficient_of_variation_max": 0.10,
                             "elastic_total_energy_relative_change_max": 0.02,
                             "energy_axis_clamps": 0,
                             "energy_monotonic_repairs": 0},
               "performance": {
                   "closure_overhead_fraction_max": 0.05,
                   "campaign_mean_closure_overhead_fraction": float(np.mean(overhead)),
                   "campaign_maximum_closure_overhead_fraction": float(np.max(overhead)),
                   "n_run_failures": int(np.count_nonzero(overhead >= 0.05)),
               },
               "artifact_sha256": (next(iter(artifact_digests))
                                   if artifact_consistent else None),
               "artifact_consistent": artifact_consistent,
               "runs": records, "cases": cases,
               "physics_gate_pass": bool(artifact_consistent and all(
                   case["physics_pass"] for case in cases
                   if case["tier"] == "gate")),
               "production_gate_pass": bool(artifact_consistent and all(
                   case["production_pass"] for case in cases
                   if case["tier"] == "gate"))}
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    gate_keys = [(1.0, 2.0), (0.95, 2.0), (0.8, 2.0),
                 (1.0, 3.0), (0.95, 3.0), (0.8, 3.0)]
    available = [key for key in gate_keys if key in grouped]
    ncols = 3
    nrows = max(1, int(np.ceil(len(available) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.5 * nrows), squeeze=False)
    colors = {0.75: "#0072B2", 1.25: "#D55E00"}
    for ax, key in zip(axes.flat, available):
        alpha, ar = key
        matching = [(row, values) for row, values in series
                    if float(row["alpha"]) == alpha and float(row["aspect_ratio"]) == ar]
        for theta0 in sorted(colors):
            replicates = [values for row, values in matching
                          if float(row["theta0"]) == theta0]
            if not replicates:
                continue
            length = min(len(values) for values in replicates)
            tau = replicates[0][:length, 0]
            stack = np.array([values[:length, 1] for values in replicates])
            mean, spread = np.mean(stack, axis=0), np.std(stack, axis=0)
            ax.plot(tau, mean, color=colors[theta0], lw=1.6,
                    label=fr"$\theta_0={theta0:g}$")
            if len(replicates) > 1:
                ax.fill_between(tau, mean - spread, mean + spread,
                                color=colors[theta0], alpha=0.14, linewidth=0)
        target_text = matching[0][0]["target_theta"] if matching else ""
        if target_text:
            target = float(target_text)
            ax.axhspan(0.9 * target, 1.1 * target, color="#009E73", alpha=0.13)
            ax.axhline(target, color="#009E73", lw=1.0, ls="--", label="DEM target")
        ax.set_title(fr"$\alpha={alpha:g}$, AR={ar:g}")
        ax.set_xlabel("collisions per particle")
        ax.set_ylabel(r"$\theta=T_{tr}/T_{rot}$")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(available):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("HCS ratio attraction: two initial energy partitions", y=0.995)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    print(f"wrote {figure} and {summary_path}; "
          f"physics_gate_pass={summary['physics_gate_pass']}; "
          f"production_gate_pass={summary['production_gate_pass']}")


if __name__ == "__main__":
    main()
