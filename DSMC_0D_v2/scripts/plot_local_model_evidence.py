#!/usr/bin/env python3
"""Combine reduced HCS and USF comparisons into one decision figure."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"uncorrected": "#D55E00", "corrected": "#0072B2"}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hcs-uncorrected", required=True)
    parser.add_argument("--hcs-corrected", required=True)
    parser.add_argument("--usf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    hcs = {"uncorrected": load(args.hcs_uncorrected),
           "corrected": load(args.hcs_corrected)}
    usf = load(args.usf)

    hcs_keys = sorted({(case["alpha"], case["aspect_ratio"])
                       for case in hcs["uncorrected"]["cases"]})
    labels = [f"$\\alpha$={alpha:g}\nAR={ar:g}" for alpha, ar in hcs_keys]
    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 9.0))

    ax = axes[0, 0]
    targets = []
    for arm, offset in (("uncorrected", -width / 2), ("corrected", width / 2)):
        lookup = {(case["alpha"], case["aspect_ratio"]): case
                  for case in hcs[arm]["cases"]}
        values = [100.0 * abs(lookup[key]["mean_theta"]
                              - lookup[key]["target_theta"])
                  / lookup[key]["target_theta"] for key in hcs_keys]
        ax.bar(x + offset, values, width, color=COLORS[arm], label=arm)
        targets = [lookup[key]["target_theta"] for key in hcs_keys]
    ax.axhline(10.0, color="black", ls="--", lw=1.2, label="10% criterion")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"HCS mean-$\theta$ relative error (%)")
    ax.set_title("HCS accuracy at six DEM-backed nodes")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=9)

    ax = axes[0, 1]
    for arm, offset in (("uncorrected", -width / 2), ("corrected", width / 2)):
        lookup = {(case["alpha"], case["aspect_ratio"]): case
                  for case in hcs[arm]["cases"]}
        values = [100.0 * lookup[key]["mean_closure_overhead_fraction"]
                  for key in hcs_keys]
        ax.bar(x + offset, values, width, color=COLORS[arm], label=arm)
    ax.axhline(15.0, color="black", ls="--", lw=1.2, label="15% criterion")
    ax.set_xticks(x, labels)
    ax.set_ylabel("closure overhead (%)")
    ax.set_title("HCS runtime cost")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    metrics = ("stress_relative_rmse", "theta_relative_rmse")
    metric_labels = ("stress RMSE", r"$\theta$ RMSE")
    mx = np.arange(len(metrics))
    for arm, offset in (("uncorrected", -width / 2), ("corrected", width / 2)):
        values = [100.0 * usf["arms"][arm][metric] for metric in metrics]
        bars = ax.bar(mx + offset, values, width, color=COLORS[arm], label=arm)
        ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values],
                     padding=3, fontsize=9)
    ax.set_xticks(mx, metric_labels)
    ax.set_ylabel("aggregate relative error (%)")
    ax.set_title("USF comparison with DEM (four cases)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    grouped: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    for run in usf["runs"]:
        grouped[(run["arm"], run["alpha"], run["aspect_ratio"])].append(
            run["out_of_domain_fraction"])
    usf_keys = sorted({(case["alpha"], case["aspect_ratio"])
                       for case in usf["cases"]})
    ux = np.arange(len(usf_keys))
    ulabels = [f"$\\alpha$={alpha:g}\nAR={ar:g}" for alpha, ar in usf_keys]
    for arm, offset in (("uncorrected", -width / 2), ("corrected", width / 2)):
        values = [100.0 * np.mean(grouped[(arm, *key)]) for key in usf_keys]
        ax.bar(ux + offset, values, width, color=COLORS[arm], label=arm)
    ax.axhline(0.1, color="black", ls="--", lw=1.2, label="0.1% criterion")
    ax.set_xticks(ux, ulabels)
    ax.set_ylabel("out-of-domain queries (%)")
    ax.set_title("USF correction-domain validity")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    figure.suptitle(
        "Reduced local evidence: angular correction is HCS-safe but not cross-flow beneficial",
        y=0.995, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0, w_pad=1.8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
