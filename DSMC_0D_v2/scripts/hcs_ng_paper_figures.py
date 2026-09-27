#!/usr/bin/env python3
"""Journal figures for the spherocylinder HCS non-Gaussian study.

One figure answers one question.  Both control parameters are ordered, so
every series uses a single-hue ordinal ramp rather than categorical hues, and
no panel carries more than three series: the campaign's cross design (alpha
swept at three aspect ratios, AR swept at four restitutions) supplies exactly
that, which is why it is laid out this way rather than as one crowded axes.
The exact elastic block (alpha = 1) is an analytic null, not another series,
so it is drawn as a neutral reference throughout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

# ---------------------------------------------------------------- palette ---
# Validated single-hue ordinal ramps (OKLab dL >= 0.06 between steps, CVD
# dE >= 8, light end >= 2:1 on the #fcfcfb surface).  Blue carries the alpha
# series, orange the aspect-ratio series; they never appear in one panel.
BLUE = {0.50: "#184f95", 0.80: "#2a78d6", 0.95: "#86b6ef"}
ORANGE = {1.50: "#9e1c00", 2.00: "#d95821", 3.00: "#ff8d63"}
PAIR = ("#2a78d6", "#eb6834")          # two conditions, not a magnitude
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a84"
NULL = "#6f6e69"                       # the analytic elastic null
SURFACE = "#ffffff"

ALPHA_SERIES = (0.50, 0.80, 0.95)
AR_SERIES = (1.50, 2.00, 3.00)
ALL_AR = (1.10, 1.20, 1.35, 1.50, 2.00, 2.50, 3.00)
ALL_ALPHA = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": "#e3e2de", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "lines.linewidth": 1.5,
    "lines.markersize": 4.5, "figure.dpi": 200,
})
SINGLE, DOUBLE = 3.4, 7.0        # journal column widths (inches)


def tidy(ax, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=6)


def plain_log(ax, which="y"):
    """Plain numerals on a log axis: 6x10^0 is noise on a 1-7 range."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(fmt if which == "y" else FuncFormatter(lambda v, _: ""))
    if which == "y":
        axis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0)))


def panel_tag(ax, tag):
    ax.text(-0.02, 1.12, tag, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="right", color=INK)


# ------------------------------------------------------------------ data ---
def load(results: Path):
    summary = json.loads((results / "summary.json").read_text())
    sci = {(c["alpha"], c["aspect_ratio"]): c for c in summary["science_cases"]}
    loss = defaultdict(list)
    for name in os.listdir(results):
        m = re.fullmatch(
            r"alpha_([\d.]+)_AR_([\d.]+)_theta0_([\d.]+)_scaled_rep_(\d+)\.json",
            name)
        if not m:
            continue
        payload = json.loads((results / name).read_text())
        audit = payload.get("collision_audit") or {}
        loss[(float(m.group(1)), float(m.group(2)))].append(
            audit.get("loss_energy_weighted") or 0.0)
    return summary, sci, {k: float(np.mean(v)) for k, v in loss.items()}


def obs(sci, alpha, ar, name, field="mean"):
    case = sci.get((alpha, ar))
    if case is None:
        return None
    entry = case["observables"].get(name)
    return None if entry is None else entry[field]


def zeta_star(sci, loss, alpha, ar):
    """-d ln T / d tau from the energy-weighted per-collision loss.

    Per unit tau there are N/2 pair events, each removing a fraction L of the
    pair's internal energy; for a Maxwellian the collision-weighted pair
    internal energy is 2(Ttr + Trot) against a total 5T = 3Ttr + 2Trot.  This
    reproduces Haff's law fitted to the freely cooling runs to 0.02-2.3 per
    cent, the residual tracking the translational kurtosis.
    """
    if (alpha, ar) not in loss:
        return None
    theta = obs(sci, alpha, ar, "theta_rot_over_tr")
    return 2.0 * loss[(alpha, ar)] * (1.0 + theta) / (3.0 + 2.0 * theta)


def smooth_sphere_zeta(alpha):
    """Cooling rate of a SMOOTH inelastic hard-sphere gas, same convention.

    Not the theta = 1 case of ``zeta_star``: a smooth sphere carries no
    rotational degrees of freedom, so both the pair internal energy and the
    total change.  Three translational DOF give a pair internal energy of
    2 Ttr against a total (3/2) T per particle, so zeta* = (2/3) L; the smooth
    hard-sphere loss is L = (1 - a^2)/2 because <(g.n)^2>/<g^2> = 1/2 over the
    collision-weighted distribution.  Hence zeta* = (1 - a^2)/3, which is
    Haff's law exactly as fitted by Hong et al. (2022, Eq. 3).
    """
    a = np.asarray(alpha, dtype=float)
    return (1.0 - a**2) / 3.0


def alpha_eff(sci, loss, alpha, ar):
    """Restitution a smooth sphere gas needs to cool at the measured rate."""
    z = zeta_star(sci, loss, alpha, ar)
    return None if z is None else float(np.sqrt(max(0.0, 1.0 - 3.0 * z)))


def ihs_a2(alpha):
    """First Sonine coefficient of smooth inelastic hard spheres (3D),
    van Noije & Ernst (1998): the no-rotation reference for a20."""
    a = np.asarray(alpha, dtype=float)
    return (16.0 * (1.0 - a) * (1.0 - 2.0 * a**2)
            / (241.0 - 177.0 * a + 30.0 * a**2 * (1.0 - a)))


def series_plot(ax, xs, ys, es, color, label, marker="o"):
    x = [a for a, b in zip(xs, ys) if b is not None]
    e = [c for c, b in zip(es, ys) if b is not None] if es else None
    y = [b for b in ys if b is not None]
    if not y:
        return
    ax.errorbar(x, y, yerr=e, color=color, marker=marker, label=label,
                capsize=0, elinewidth=1.0, markeredgecolor=SURFACE,
                markeredgewidth=0.6, zorder=3)


# ----------------------------------------------------------------- fig 1 ---
def fig_attractor(results: Path, out: Path):
    """Both initial branches reach one attractor before the retained window."""
    runs = defaultdict(lambda: defaultdict(list))
    for name in os.listdir(results):
        m = re.fullmatch(
            r"alpha_([\d.]+)_AR_([\d.]+)_theta0_([\d.]+)_scaled_rep_(\d+)\.txt",
            name)
        if m:
            runs[(float(m.group(1)), float(m.group(2)))][
                float(m.group(3))].append(results / name)
    picks = [(0.50, 1.20), (0.80, 1.50), (0.95, 3.00)]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.25), sharex=True)
    for ax, key, tag in zip(axes, picks, "abc"):
        branches = sorted(runs[key])
        for th0, color in zip(branches, PAIR):
            curves = [np.loadtxt(p) for p in sorted(runs[key][th0])]
            n = min(len(c) for c in curves)
            tau = curves[0][:n, 1]
            theta = np.mean([c[:n, 3] / c[:n, 2] for c in curves], axis=0)
            ax.plot(np.maximum(tau, 1.0), theta, color=color, lw=1.1,
                    label=rf"$\theta_0={th0:g}$")
        ax.axvspan(500, 1500, color="#f0efec", zorder=0)
        ax.axvline(130, color=NULL, ls=(0, (4, 2)), lw=1.0, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, 1500)
        plain_log(ax, "y"); plain_log(ax, "x")
        ax.set_xlabel(r"$\tau$  (collisions per particle)")
        tidy(ax, rf"$\alpha={key[0]:g}$,  $\mathrm{{AR}}={key[1]:g}$")
        panel_tag(ax, tag)
        if ax is axes[0]:
            ax.set_ylabel(r"$T_{\rm rot}/T_{\rm tr}$")
            ax.legend(loc="lower left", handlelength=1.4)
            ax.annotate(r"$\tau=130$", xy=(130, 0.06), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=7.0, color=INK2, ha="right", va="bottom")
    axes[2].annotate("retained\nwindow", xy=(866, 0.90), xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=7.2, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig1_attractor")


# ----------------------------------------------------------------- fig 2 ---
def fig_equipartition(sci, out: Path):
    """How far the HCS sits from equipartition, and how it diverges at AR->1."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    ax = axes[0]
    for a in ALPHA_SERIES:
        ys = [obs(sci, a, ar, "theta_rot_over_tr") for ar in ALL_AR]
        es = [obs(sci, a, ar, "theta_rot_over_tr", "replicate_stderr") for ar in ALL_AR]
        series_plot(ax, ALL_AR, ys, es, BLUE[a], rf"$\alpha={a:g}$")
    ax.axhline(1.0, color=NULL, ls=(0, (4, 2)), lw=1.1, zorder=2)
    ax.annotate(r"equipartition ($\alpha=1$ exactly)", xy=(3.15, 1.0),
                xytext=(0, 5), textcoords="offset points",
                fontsize=7.2, color=INK2, va="bottom", ha="right")
    ax.set_yscale("log"); plain_log(ax)
    ax.set_xlabel("aspect ratio")
    ax.set_ylabel(r"$\theta^{H}=T_{\rm rot}/T_{\rm tr}$")
    ax.set_xlim(1.0, 3.2)
    ax.legend(loc="upper right", handlelength=1.4)
    tidy(ax, "Equipartition breaking"); panel_tag(ax, "a")

    ax = axes[1]
    slopes = []
    for a in ALPHA_SERIES:
        x = np.array([ar - 1.0 for ar in ALL_AR])
        y = np.array([obs(sci, a, ar, "theta_rot_over_tr") for ar in ALL_AR],
                     dtype=object)
        keep = np.array([v is not None for v in y])
        xx = x[keep]; yy = np.array([v for v in y if v is not None]) - 1.0
        ax.plot(xx, yy, color=BLUE[a], marker="o", label=rf"$\alpha={a:g}$",
                markeredgecolor=SURFACE, markeredgewidth=0.6)
        if len(xx) >= 3:
            sl = np.polyfit(np.log(xx[:3]), np.log(yy[:3]), 1)[0]
            slopes.append((a, sl))
    for k, (a, sl) in enumerate(slopes):
        ax.text(0.03, 0.30 - 0.09 * k, rf"$\alpha={a:g}$:  slope ${sl:.2f}$",
                transform=ax.transAxes, fontsize=7.4, color=BLUE[a],
                va="top", ha="left")
    ax.text(0.03, 0.39, "near-sphere fit", transform=ax.transAxes,
            fontsize=7.4, color=INK2, va="top", ha="left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.08, 2.6)
    ax.set_xlabel(r"$\mathrm{AR}-1$"); ax.set_ylabel(r"$\theta^{H}-1$")
    tidy(ax, "Divergence into the smooth-sphere limit")
    panel_tag(ax, "b")
    fig.tight_layout()
    save(fig, out, "fig2_equipartition")


# ----------------------------------------------------------------- fig 3 ---
def fig_alpha_eff(sci, loss, out: Path):
    """Elongation makes the gas cool like a less inelastic sphere gas."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    ax = axes[0]
    grid = np.linspace(0.45, 1.0, 50)
    ax.plot(grid, grid, color=NULL, ls=(0, (4, 2)), lw=1.1, zorder=2)
    ax.text(0.79, 0.745, "smooth spheres", fontsize=7.4, color=INK2,
            rotation=36, ha="center", va="center")
    for ar in AR_SERIES:
        ys = [alpha_eff(sci, loss, a, ar) for a in ALL_ALPHA]
        series_plot(ax, ALL_ALPHA, ys, None, ORANGE[ar],
                    rf"$\mathrm{{AR}}={ar:g}$")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\alpha_{\rm eff}$")
    ax.set_xlim(0.45, 1.02); ax.set_ylim(0.45, 1.02)
    ax.legend(loc="upper left", handlelength=1.4)
    tidy(ax, "Shape shields dissipation"); panel_tag(ax, "a")

    ax = axes[1]
    for a in ALPHA_SERIES:
        ys = [(alpha_eff(sci, loss, a, ar) - a
               if alpha_eff(sci, loss, a, ar) is not None else None)
              for ar in ALL_AR]
        series_plot(ax, ALL_AR, ys, None, BLUE[a], rf"$\alpha={a:g}$")
    ax.axhline(0.0, color=NULL, ls=(0, (4, 2)), lw=1.1)
    ax.set_xlabel("aspect ratio")
    ax.set_ylabel(r"$\alpha_{\rm eff}-\alpha$")
    ax.set_xlim(1.0, 3.2)
    ax.legend(loc="center right", handlelength=1.4)
    tidy(ax, r"Shielding saturates near AR $\approx$ 2"); panel_tag(ax, "b")
    fig.tight_layout()
    save(fig, out, "fig3_alpha_eff")


# ----------------------------------------------------------------- fig 4 ---
def fig_cumulants(sci, out: Path):
    """The three fourth-order cumulants, along each arm of the cross design."""
    rows = (("a20", r"$a_{20}^{H}$"), ("a02", r"$a_{02}^{H}$"),
            ("a11", r"$a_{11}^{H}$"))
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 4.5))
    for j, (name, label) in enumerate(rows):
        ax = axes[0, j]
        for a in ALPHA_SERIES:
            ys = [obs(sci, a, ar, name) for ar in ALL_AR]
            es = [obs(sci, a, ar, name, "replicate_stderr") for ar in ALL_AR]
            series_plot(ax, ALL_AR, ys, es, BLUE[a], rf"$\alpha={a:g}$")
        ax.axhline(0.0, color=NULL, ls=(0, (4, 2)), lw=1.0)
        ax.set_xlabel("aspect ratio"); ax.set_ylabel(label)
        ax.set_xlim(1.0, 3.2)
        tidy(ax); panel_tag(ax, "abc"[j])
        if j == 0:
            ax.legend(loc="upper right", handlelength=1.4)

        ax = axes[1, j]
        for ar in AR_SERIES:
            ys = [obs(sci, a, ar, name) for a in ALL_ALPHA]
            es = [obs(sci, a, ar, name, "replicate_stderr") for a in ALL_ALPHA]
            series_plot(ax, ALL_ALPHA, ys, es, ORANGE[ar],
                        rf"$\mathrm{{AR}}={ar:g}$")
        ax.axhline(0.0, color=NULL, ls=(0, (4, 2)), lw=1.0)
        if name == "a20":
            g = np.linspace(0.5, 1.0, 120)
            ax.plot(g, ihs_a2(g), color=NULL, ls=(0, (1, 1.6)), lw=1.2,
                    zorder=2, label="smooth IHS (Sonine)")
        ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(label)
        tidy(ax); panel_tag(ax, "def"[j])
        if j == 0:
            ax.legend(loc="upper right", handlelength=1.4, ncol=1)
    fig.tight_layout()
    save(fig, out, "fig4_cumulants")


# ----------------------------------------------------------------- fig 5 ---
def fig_alignment(sci, out: Path):
    """Observables with no rough-sphere analogue: the body axis enters."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5))
    ax = axes[0]
    for a in ALPHA_SERIES:
        ys = [obs(sci, a, ar, "A_cu") for ar in ALL_AR]
        es = [obs(sci, a, ar, "A_cu", "replicate_stderr") for ar in ALL_AR]
        series_plot(ax, ALL_AR, ys, es, BLUE[a], rf"$\alpha={a:g}$")
    ax.axhline(0.0, color=NULL, ls=(0, (4, 2)), lw=1.0)
    ax.annotate(r"$\alpha=1$ exactly", xy=(2.3, 0.0), xytext=(0, 4),
                textcoords="offset points", fontsize=7.2, color=INK2)
    ax.set_xlabel("aspect ratio"); ax.set_xlim(1.0, 3.2)
    ax.set_ylabel(r"$A_{cu}$")
    ax.legend(loc="upper left", handlelength=1.4)
    tidy(ax, "Velocity\u2013axis alignment"); panel_tag(ax, "a")

    ax = axes[1]
    for a in ALPHA_SERIES:
        ys = [obs(sci, a, ar, "A_cw_quadrupolar") for ar in ALL_AR]
        series_plot(ax, ALL_AR, ys, None, BLUE[a], rf"$\alpha={a:g}$")
        geom = [(-0.5 * obs(sci, a, ar, "A_cu")
                 if obs(sci, a, ar, "A_cu") is not None else None)
                for ar in ALL_AR]
        x = [ar for ar, g in zip(ALL_AR, geom) if g is not None]
        ax.plot(x, [g for g in geom if g is not None], color=BLUE[a],
                ls=(0, (1, 1.6)), lw=1.2, marker=None)
    ax.axhline(0.0, color=NULL, ls=(0, (4, 2)), lw=1.0)
    ax.set_xlabel("aspect ratio"); ax.set_xlim(1.0, 3.2)
    ax.set_ylabel(r"$A_{cw}$")
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.5, label="measured"),
                       Line2D([], [], color=INK2, lw=1.2, ls=(0, (1, 1.6)),
                              label=r"axis geometry, $-A_{cu}/2$")],
              loc="lower left", handlelength=1.6)
    tidy(ax, "Velocity\u2013spin alignment"); panel_tag(ax, "b")

    ax = axes[2]
    for a in ALPHA_SERIES:
        ys = []
        for ar in ALL_AR:
            acu, acw = obs(sci, a, ar, "A_cu"), obs(sci, a, ar, "A_cw_quadrupolar")
            ys.append(None if acu is None else (acw + acu / 2.0) / acw)
        series_plot(ax, ALL_AR, ys, None, BLUE[a], rf"$\alpha={a:g}$")
    ax.set_xlabel("aspect ratio"); ax.set_xlim(1.0, 3.2); ax.set_ylim(0, 1)
    ax.set_ylabel(r"$1+A_{cu}/2A_{cw}$")
    tidy(ax, "Dynamical share of $A_{cw}$"); panel_tag(ax, "c")
    fig.tight_layout()
    save(fig, out, "fig5_alignment")


# ----------------------------------------------------------------- fig 6 ---
def fig_tails(sci, out: Path):
    """Every fitted tail exponent against its exact-Maxwellian null."""
    chans = (("c", r"$\gamma_c$", r"$\phi_c(c)\sim e^{-\gamma_c c}$"),
             ("w", r"$\gamma_w$", r"$\phi_w(w)\sim w^{-\gamma_w}$"),
             ("x", r"$\gamma_{cw}$", r"$\phi_{cw}\sim x^{-\gamma_{cw}}$"))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5))
    for ax, (nm, lab, form), tag in zip(axes, chans, "abc"):
        null = [c["tail_fits"][nm]["gamma"] for c in sci.values()
                if c["alpha"] >= 1.0 and c["tail_fits"][nm].get("fit_ready")]
        mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
        ax.axhspan(mu - 3 * sd, mu + 3 * sd, color="#eceae4", zorder=0)
        ax.axhline(mu, color=NULL, ls=(0, (4, 2)), lw=1.1, zorder=2)
        for ar in AR_SERIES:
            ys = []
            for a in ALL_ALPHA:
                c = sci.get((a, ar))
                f = None if c is None else c["tail_fits"][nm]
                ys.append(f["gamma"] if f and f.get("fit_ready") else None)
            series_plot(ax, ALL_ALPHA, ys, None, ORANGE[ar],
                        rf"$\mathrm{{AR}}={ar:g}$")
        ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(lab)
        tidy(ax, form); panel_tag(ax, tag)
        if tag == "a":
            ax.legend(loc="lower right", handlelength=1.4)
            ax.annotate("exact Maxwellian ($\\alpha=1$), $\\pm3\\sigma$",
                        xy=(0.955, mu), xytext=(0, 7), textcoords="offset points",
                        fontsize=7.2, color=INK2, va="bottom", ha="right")
    fig.tight_layout()
    save(fig, out, "fig6_tails")


def save(fig, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out/stem}.pdf and .png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/hcs_ng_sweep_angular_v8")
    p.add_argument("--output", default="reports/figures/hcs_ng_v8")
    args = p.parse_args()
    results, out = Path(args.results), Path(args.output)
    summary, sci, loss = load(results)
    if not summary.get("scientific_outputs_released", False):
        raise SystemExit("sweep summary has not released scientific outputs")
    fig_attractor(results, out)
    fig_equipartition(sci, out)
    fig_alpha_eff(sci, loss, out)
    fig_cumulants(sci, out)
    fig_alignment(sci, out)
    fig_tails(sci, out)


if __name__ == "__main__":
    main()
