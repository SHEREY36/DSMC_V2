#!/usr/bin/env python3
"""Plot full-domain excitation evidence and the selective release decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--offline-validation", required=True)
    parser.add_argument("--coefficient-rows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    offline = json.loads(Path(args.offline_validation).read_text())
    coefficients = json.loads(Path(args.coefficient_rows).read_text())
    nodes = summary["nodes"]
    rows = coefficients["coefficient_rows"]

    alphas = sorted({float(node["coordinates"][0]) for node in nodes})
    thetas = sorted({float(node["coordinates"][1]) for node in nodes})
    ars = sorted({float(node["coordinates"][2]) for node in nodes})
    status = {(float(node["coordinates"][0]),
               float(node["coordinates"][1]),
               float(node["coordinates"][2])):
              (2 if node["response_fit_ready"] else
               1 if node["screening_pass"] else 0)
              for node in nodes}

    figure = plt.figure(figsize=(15.0, 10.0))
    outer = figure.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    heatmaps = outer[0, 0].subgridspec(2, 2, hspace=0.58, wspace=0.16)
    cmap = ListedColormap(("#D55E00", "#E69F00", "#009E73"))
    cmap.set_bad("#E5E5E5")
    norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), cmap.N)
    image = None
    heat_axes = []
    for index, alpha in enumerate(alphas):
        ax = figure.add_subplot(heatmaps[index // 2, index % 2])
        heat_axes.append(ax)
        grid = np.ma.masked_invalid(np.asarray([
            [status.get((alpha, theta, ar), np.nan) for ar in ars]
            for theta in thetas]))
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(fr"$\alpha={alpha:g}$")
        ax.set_xticks(range(len(ars)), [f"{value:g}" for value in ars], rotation=45)
        ax.set_yticks(range(len(thetas)), [f"{value:g}" for value in thetas])
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("aspect ratio")
        if index % 2:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"$\theta$")
    heat_axes[0].text(
        0.0, 1.20, "gray = unsampled physical combination",
        transform=heat_axes[0].transAxes, fontsize=8, color="#555555")
    heat_axes[1].legend(
        handles=[Patch(color=color, label=label) for color, label in zip(
            ("#D55E00", "#E69F00", "#009E73"),
            ("central fail", "screen only", "fit ready"))],
        loc="upper right", bbox_to_anchor=(1.0, 1.30), ncol=3,
        frameon=False, fontsize=8)

    ax = figure.add_subplot(outer[0, 1])
    central = offline["central_trust_region"]
    held_energy = offline["conditional_energy_quantile_wasserstein1"]
    held_angle = offline["angular_quantile_wasserstein1"]
    labels = ("central\nenergy", "held-out\nenergy",
              "central\nangular", "held-out\nangular")
    baseline = (
        central["conditional_energy_quantile_wasserstein1"]["baseline"]["p95"],
        held_energy["baseline"]["p95"],
        central["angular_quantile_wasserstein1"]["baseline"]["p95"],
        held_angle["baseline"]["p95"],
    )
    corrected = (
        central["conditional_energy_quantile_wasserstein1"]
        ["artifact_tangent"]["p95"],
        held_energy["artifact_tangent"]["p95"],
        central["angular_quantile_wasserstein1"]["multivariate"]["p95"],
        held_angle["multivariate"]["p95"],
    )
    x = np.arange(len(labels))
    ax.bar(x - 0.19, baseline, 0.38, color="#777777", label="uncorrected")
    ax.bar(x + 0.19, corrected, 0.38, color="#0072B2", label="fitted response")
    ax.set_xticks(x, labels)
    ax.set_ylabel("95th-percentile quantile $W_1$")
    ax.set_title("Distribution response: energy regresses, angle improves")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    ax = figure.add_subplot(outer[1, 0])
    rmse = np.sort(np.asarray([
        node["maximum_full_heldout_relative_rmse"]
        for node in offline["node_validation"]], dtype=float))
    ax.plot(np.arange(1, len(rmse) + 1), rmse, color="#0072B2", lw=1.8)
    ax.axhline(0.15, color="#D55E00", ls="--", lw=1.4,
               label="linearity tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("physical nodes, sorted")
    ax.set_ylabel("maximum held-out relative RMSE")
    ax.set_title(f"Natural-parameter response ({np.count_nonzero(rmse > 0.15)}/144 above tolerance)")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)

    ax = figure.add_subplot(outer[1, 1])
    release = np.asarray([row["angular_release"] for row in rows], dtype=bool)
    trusts = np.asarray([row["trust_amplitude"] for row in rows], dtype=float)
    categories = (r"angular released" "\n" r"$|\eta|\leq0.5$",
                  "fully suppressed\ncentral sentinel fail",
                  "energy released")
    values = (int(np.count_nonzero(release & (trusts >= 0.5))),
              int(np.count_nonzero(~release)),
              int(np.count_nonzero([
                  np.any(np.asarray(row["beta_deployed"], dtype=bool)[:4])
                  for row in rows])))
    bars = ax.bar(categories, values, color=("#009E73", "#D55E00", "#0072B2"))
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, 152)
    ax.set_ylabel("physical nodes")
    ax.set_title("Conservative evidence-artifact release")
    ax.grid(axis="y", alpha=0.2)

    figure.suptitle(
        "Full-domain excitation campaign: 15,552 completed observations, 144 physical nodes",
        y=0.995, fontsize=14)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
