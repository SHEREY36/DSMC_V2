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


def analyse(row: dict[str, str]) -> tuple[dict, np.ndarray]:
    prefix = row["output_prefix"]
    data = np.atleast_2d(np.loadtxt(Path(prefix + ".txt")))
    diagnostics = json.loads(Path(prefix + ".json").read_text())
    tau, ttr, trot, total = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    theta = ttr / trot
    start = max(1, int(0.7 * len(theta)))
    x, y = tau[start:], theta[start:]
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
    mean = float(np.mean(y))
    span = float(x[-1] - x[0]) if len(x) > 1 else 0.0
    drift = abs(slope) * span / max(abs(mean), 1.0e-12)
    target = None if not row["target_theta"] else float(row["target_theta"])
    result = {
        "task_id": int(row["task_id"]), "tier": row["tier"],
        "alpha": float(row["alpha"]), "aspect_ratio": float(row["aspect_ratio"]),
        "theta0": float(row["theta0"]), "target_theta": target,
        "late_theta_mean": mean,
        "late_theta_std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        "late_relative_drift": drift,
        "relative_target_error": None if target is None else abs(mean - target) / target,
        "temperature_cools": bool(total[-1] < total[0] and np.all(total > 0.0)),
        "bounded": bool(np.all(np.isfinite(theta)) and np.all(theta > 0.0)
                        and np.max(theta) < 10.0),
        "runtime_gate_pass": bool(diagnostics["runtime_gate"]["pass"]),
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
        convergence = ((float(np.ptp(means)) / max(float(np.mean(means)), 1.0e-12))
                       if len(means) > 1 else None)
        gate = (len(means) > 1 and convergence <= 0.10
                and all(item["bounded"] and item["temperature_cools"]
                        and item["runtime_gate_pass"]
                        and item["late_relative_drift"] <= 0.10 for item in items)
                and (target is None or abs(float(np.mean(means)) - target) / target <= 0.10))
        cases.append({"alpha": alpha, "aspect_ratio": ar, "tier": items[0]["tier"],
                      "target_theta": target, "mean_theta": float(np.mean(means)),
                      "initial_condition_spread": convergence, "pass": bool(gate)})

    summary = {"criteria": {"target_relative_error_max": 0.10,
                             "late_relative_drift_max": 0.10,
                             "initial_condition_spread_max": 0.10},
               "runs": records, "cases": cases,
               "gate_pass": all(case["pass"] for case in cases if case["tier"] == "gate")}
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
        for row, values in matching:
            theta0 = float(row["theta0"])
            ax.plot(values[:, 0], values[:, 1], color=colors[theta0], lw=1.4,
                    label=fr"$\theta_0={theta0:g}$")
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
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("HCS ratio attraction: two initial energy partitions", y=1.01)
    fig.tight_layout()
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    print(f"wrote {figure} and {summary_path}; gate_pass={summary['gate_pass']}")


if __name__ == "__main__":
    main()
