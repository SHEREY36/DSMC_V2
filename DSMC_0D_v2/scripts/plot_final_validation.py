#!/usr/bin/env python3
"""Final-model validation figures and tables against DEM (USF and HCS).

    hpc/python.sh DSMC_0D_v2/scripts/plot_final_validation.py --tag final_v1_val \
        --output results/validation/dem_comparison/final_v1

USF: relative difference from DEM of T*, theta, P*_xy and N1 for the legacy
model, the alpha-refined model and the final model.  HCS: steady theta* of the
final model, DEM and (where the node grid resolves it) exact CTC, and the
relative difference from DEM.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLORS = {1.1: "#4c78a8", 1.25: "#72b7b2", 1.5: "#e76f51", 2.0: "#1b9e77", 2.5: "#d99a00", 3.0: "#9b59b6"}
MARKERS = {1.1: "o", 1.25: "P", 1.5: "s", 2.0: "D", 2.5: "^", 3.0: "v"}
# exact-CTC HCS roots where the theta grid resolves them (zero of d ln(theta)/d tau
# on Maxwellian node pairs; reports/DSMC_CLOSURE_FORMULATION_FINAL.md, Section 7)
CTC_HCS = {(1.1, 0.5): 0.0210, (1.1, 0.55): 0.0338, (1.1, 0.6): 0.0453, (1.1, 0.65): 0.0600,
           (1.1, 0.7): 0.0787, (1.1, 0.75): 0.0981, (1.1, 0.8): 0.1315, (1.1, 0.85): 0.1842,
           (1.5, 0.5): 0.3592}


def usf_table(path):
    return {(float(r["AR"]), round(float(r["alpha"]), 2)): r for r in csv.DictReader(open(path))
            if r["arm"] == "encounter_memory"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="final_v1_val")
    parser.add_argument("--previous", default="alpha_v1_20261001")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    final = usf_table(f"results/validation/usf/usf_encounter_{args.tag}/usf_vs_dem.csv")
    previous = usf_table(f"results/validation/usf/usf_encounter_{args.previous}/usf_vs_dem.csv")

    quantities = (("Tstar", r"$T^*$"), ("theta", r"$\theta=T_{tr}/T_{rot}$"),
                  ("Pk_xy", r"$|P^{k*}_{xy}|$"), ("N1k", r"$N_1^{k*}$"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for ax, (key, label) in zip(axes.ravel(), quantities):
        ax.axhspan(-5, 5, color="0.92", zorder=0)
        for ar in (1.5, 2.0, 2.5, 3.0):
            keys = sorted(k for k in final if k[0] == ar)
            a = [k[1] for k in keys]
            sign = -1.0 if key == "Pk_xy" else 1.0       # P_xy < 0: plot |DSMC|/|DEM| - 1
            leg = [100 * sign * float(final[k]["legacy_" + key + "_vs_DEM"]) for k in keys]
            old = [100 * sign * float(previous[k][key + "_vs_DEM"]) for k in keys]
            new = [100 * sign * float(final[k][key + "_vs_DEM"]) for k in keys]
            if key == "Pk_xy":
                leg, old, new = ([-v for v in s] for s in (leg, old, new))
            ax.plot(a, leg, ls=":", color=COLORS[ar], alpha=0.45, lw=1.0)
            ax.plot(a, old, ls="--", marker=MARKERS[ar], mfc="none", color=COLORS[ar], alpha=0.7, lw=1.0, ms=5)
            ax.plot(a, new, ls="-", marker=MARKERS[ar], color=COLORS[ar], lw=1.7, ms=5.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(f"DSMC vs DEM, {label} (%)")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylim(-30, 12); axes[1, 1].set_ylim(-32, 8)
    for ax in axes[1]:
        ax.set_xlabel(r"restitution $\alpha$")
    handles = [Line2D([], [], color=COLORS[ar], marker=MARKERS[ar], label=f"AR={ar:g}") for ar in (1.5, 2.0, 2.5, 3.0)]
    handles += [Line2D([], [], color="0.4", ls=":", label="legacy model"),
                Line2D([], [], color="0.4", ls="--", marker="o", mfc="none", label=r"$\alpha$-refined"),
                Line2D([], [], color="0.4", ls="-", marker="o", label="final model")]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("USF: relative difference from DEM (grey band: 5 %)", y=1.04)
    fig.tight_layout()
    fig.savefig(out / "usf_error_by_model.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    hcs = {(float(r["AR"]), round(float(r["alpha"]), 2)): r for r in
           csv.DictReader(open(f"results/validation/hcs/hcs_encounter_{args.tag}/hcs_vs_dem.csv"))}
    dem = {(float(r["AR"]), round(float(r["alpha"]), 2)): float(r["theta_star"])
           for r in csv.DictReader(open("DSMC_0D_v2/reference/hcs_dem_fresh_v1.csv"))}
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))
    for ar in sorted({k[0] for k in dem}):
        keys = sorted(k for k in dem if k[0] == ar and k in hcs)
        a = [k[1] for k in keys]
        left.plot(a, [dem[k] for k in keys], "-", color=COLORS[ar], marker=MARKERS[ar], ms=5, lw=1.4)
        left.plot(a, [float(hcs[k]["theta"]) for k in keys], "--", color=COLORS[ar], marker=MARKERS[ar],
                  mfc="white", ms=6, lw=1.2)
        right.plot(a, [100 * float(hcs[k]["vs_DEM"]) for k in keys], "-", color=COLORS[ar],
                   marker=MARKERS[ar], ms=5, lw=1.4)
        ck = sorted(k for k in CTC_HCS if k[0] == ar)
        if ck:
            left.plot([k[1] for k in ck], [CTC_HCS[k] for k in ck], "*", color=COLORS[ar], ms=11,
                      markeredgecolor="k", markeredgewidth=0.6, ls="none")
            right.plot([k[1] for k in ck], [100 * (CTC_HCS[k] / dem[k] - 1) for k in ck], "*",
                       color=COLORS[ar], ms=11, markeredgecolor="k", markeredgewidth=0.6, ls="none")
    left.set_yscale("log"); left.set_ylabel(r"HCS $\theta^*=T_{tr}/T_{rot}$")
    right.axhspan(-5, 5, color="0.92", zorder=0); right.axhline(0, color="k", lw=0.8)
    right.set_ylabel("relative difference from DEM (%)")
    for ax in (left, right):
        ax.set_xlabel(r"restitution $\alpha$"); ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=COLORS[ar], marker=MARKERS[ar], label=f"AR={ar:g}") for ar in sorted(COLORS)]
    handles += [Line2D([], [], color="0.3", marker="o", label="DEM"),
                Line2D([], [], color="0.3", ls="--", marker="o", mfc="white", label="DSMC, final model"),
                Line2D([], [], color="0.5", marker="*", ls="none", ms=10, markeredgecolor="k", label="exact CTC (binary)")]
    fig.legend(handles=handles, loc="upper center", ncol=9, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "hcs_theta_final.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    summary = {}
    for name, table in (("final", final), ("alpha_refined", previous)):
        summary[name] = {q: {"median_abs": float(np.median([abs(float(table[k][q + "_vs_DEM"])) for k in final])),
                             "max_abs": float(np.max([abs(float(table[k][q + "_vs_DEM"])) for k in final]))}
                         for q, _ in quantities}
    errs = np.array([abs(float(r["vs_DEM"])) for k, r in hcs.items() if k[1] < 1.0])
    summary["hcs_final"] = {"coordinates": int(len(errs)), "median_abs": float(np.median(errs)),
                            "max_abs": float(errs.max()), "above_5pct": int((errs > 0.05).sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
