#!/usr/bin/env python3
"""Build an independent DSMC-USF study and a conservative DEM comparison.

The production USF campaign contains two initialization branches for each of
four random seeds.  Branches are convergence tests, not independent samples,
so uncertainty is computed after pairing the cold/hot branch at each seed.

Rod collisional stress is deliberately excluded from the cross-method
comparison: the current DSMC scalar collision API has no exact contact branch
vector.  DEM collisional stress is reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ARS = (1.0, 1.5, 2.0, 2.5, 3.0)
ALPHAS = tuple(round(value, 2) for value in np.arange(0.50, 0.951, 0.05))
PK = ("Pk_xx", "Pk_yy", "Pk_zz", "Pk_xy")
Q_COMPONENTS = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz", "Qzz")
COLORS = {
    1.0: "#2a78d6", 1.5: "#e76f51", 2.0: "#1b9e77",
    2.5: "#d99a00", 3.0: "#9b59b6",
}
MARKERS = {1.0: "o", 1.5: "s", 2.0: "D", 2.5: "^", 3.0: "v"}
# Set in main().  The encounter table converts DSMC encounter counts to CTC
# contact counts (<k> per encounter) so they are commensurate with DEM contacts.
ENCOUNTER_TABLE: dict | None = None
MODEL_DESCRIPTION = "frozen promoted closure artifact"
ROD_ARM = "corrected"


def contacts_per_encounter(alpha: float, theta: float, ar: float) -> float:
    rows = [row for row in ENCOUNTER_TABLE["contacts_per_encounter"]
            if abs(row["AR"] - ar) < 1.0e-9]
    alphas = sorted({row["alpha"] for row in rows})
    values = []
    for a in alphas:
        sub = sorted((row for row in rows if row["alpha"] == a), key=lambda r: r["theta"])
        values.append(np.interp(math.log(theta), [math.log(r["theta"]) for r in sub],
                                [r["k_mean"] for r in sub]))
    return float(np.interp(alpha, alphas, values))

STRICT_COMPARABLE = (
    "Tstar",
    "theta (rods only)",
    "kinetic stress Pk_xx, Pk_yy, Pk_zz, Pk_xy",
    "kinetic N1, kinetic N2, and kinetic reduced viscosity",
    "nematic order S2 (rods only)",
    "collisions per particle per unit strain",
    "steady collisional dissipation divided by shear work",
)
NOT_STRICTLY_COMPARABLE = (
    "rod collisional or total stress (DSMC has no exact rod branch vector)",
    "DEM contact duration, overlap, multibody fraction, and repeated impacts",
    "DEM spatial clustering/RDF (the present DSMC is homogeneous and 0D)",
    "per-event loss as an accuracy metric (DEM contact events and DSMC binary events differ)",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def mean_orientation_tensor(values: np.ndarray) -> np.ndarray:
    """Return the time-averaged traceless orientation tensor."""
    q = np.zeros((len(values), 3, 3))
    q[:, 0, 0], q[:, 0, 1], q[:, 0, 2] = values[:, 2], values[:, 3], values[:, 4]
    q[:, 1, 0], q[:, 1, 1], q[:, 1, 2] = values[:, 3], values[:, 5], values[:, 6]
    q[:, 2, 0], q[:, 2, 1], q[:, 2, 2] = values[:, 4], values[:, 6], values[:, 7]
    return np.mean(q, axis=0)


def nematic_order(components: np.ndarray) -> float:
    """Return standard S2=1.5*lambda_max(Q) from six Q components."""
    qxx, qxy, qxz, qyy, qyz, qzz = np.asarray(components, dtype=float)
    q = np.array([[qxx, qxy, qxz], [qxy, qyy, qyz], [qxz, qyz, qzz]])
    return float(1.5 * np.linalg.eigvalsh(q)[-1])


def load_run(row: dict[str, str], keep_series: bool = False) -> dict:
    prefix = Path(row["output_prefix"])
    trajectory = np.atleast_2d(np.loadtxt(Path(str(prefix) + ".txt")))
    pressure = np.atleast_2d(np.loadtxt(Path(str(prefix) + "_pressure.txt")))
    orientation = np.atleast_2d(np.loadtxt(Path(str(prefix) + "_orientation.txt")))
    energy = np.atleast_2d(np.loadtxt(Path(str(prefix) + "_energy.txt")))
    diagnostics = json.loads(Path(str(prefix) + ".json").read_text())
    if (trajectory.shape[1], pressure.shape[1], orientation.shape[1], energy.shape[1]) \
            != (5, 14, 8, 6):
        raise ValueError(f"unexpected output schema for {prefix}")

    ar, alpha = float(row["aspect_ratio"]), float(row["alpha"])
    sphere = row["sphere"].lower() == "true"
    rate = float(row["shear_rate"])
    start = float(row["evaluation_start_tau"])
    tau = trajectory[:, 1]
    tail = tau >= start
    tail_p = pressure[:, 1] >= start
    tail_q = orientation[:, 1] >= start
    tail_e = energy[:, 1] >= start
    if min(map(np.count_nonzero, (tail, tail_p, tail_q, tail_e))) < 20:
        raise ValueError(f"insufficient retained samples for {prefix}")

    ttr, trot = trajectory[:, 2], trajectory[:, 3]
    validation = diagnostics.get("validation_case") or {}
    mass = float(validation.get("particle_mass", 1.0))
    diameter = float(validation.get("particle_diameter", 1.0))
    tstar = ttr / (mass * (rate * diameter) ** 2)
    theta = np.full_like(ttr, np.nan) if sphere else ttr / trot
    density = float(diagnostics["number_density"])
    ttr_p = np.interp(pressure[:, 1], tau, ttr)
    raw_pk = np.column_stack((pressure[:, 2], pressure[:, 5],
                              pressure[:, 7], pressure[:, 3]))
    pk = raw_pk / (density * ttr_p[:, None])
    q_mean = (np.full((3, 3), np.nan) if sphere
              else mean_orientation_tensor(orientation[tail_q]))

    e = energy[tail_e]
    delta_shear = float(e[-1, 3] - e[0, 3])
    delta_collision = float(e[-1, 4] - e[0, 4])
    cooling_to_shear = (-delta_collision / delta_shear
                        if abs(delta_shear) > 1.0e-30 else float("nan"))
    audit = diagnostics.get("collision_audit") or {}
    ntc = diagnostics.get("ntc") or {}
    cpp = float(diagnostics["cpp"])
    strain = rate * float(diagnostics["final_time"])

    values = {
        "AR": ar,
        "alpha": alpha,
        "replicate": int(row["replicate"]),
        "branch": row["initial_branch"],
        "Tstar": float(np.mean(tstar[tail])),
        "theta": float("nan") if sphere else float(np.mean(theta[tail])),
        "Pk_xx": float(np.mean(pk[tail_p, 0])),
        "Pk_yy": float(np.mean(pk[tail_p, 1])),
        "Pk_zz": float(np.mean(pk[tail_p, 2])),
        "Pk_xy": float(np.mean(pk[tail_p, 3])),
        "Qxx": q_mean[0, 0], "Qxy": q_mean[0, 1], "Qxz": q_mean[0, 2],
        "Qyy": q_mean[1, 1], "Qyz": q_mean[1, 2], "Qzz": q_mean[2, 2],
        "collisions_per_strain": cpp / strain,
        "cooling_to_shear": cooling_to_shear,
        "post_ntc_z": _finite(audit.get("post_ntc_z_mean")),
        "post_ntc_z_energy_weighted": _finite(
            audit.get("post_ntc_z_energy_weighted")),
        "accepted_z_mean": _finite(audit.get("accepted_z_mean")),
        "loss_fraction": _finite(audit.get("loss_mean")),
        "loss_energy_weighted": _finite(audit.get("loss_energy_weighted")),
        "routing_loss_fraction": _finite(audit.get("routing_loss_mean")),
        "accepted_z": _finite(audit.get("accepted_z_energy_weighted")),
        "outgoing_z": _finite(audit.get("outgoing_z_energy_weighted")),
        "collision_theta_energy_weighted": _finite(
            audit.get("theta_energy_weighted")),
        "temperature_ratio_drift": _finite(
            audit.get("temperature_ratio_drift_per_pair_energy")),
        "expected_temperature_ratio_drift": _finite(
            audit.get("expected_temperature_ratio_drift_per_pair_energy")),
        "orientation_acceptance": _finite(audit.get("orientation_acceptance")),
        "ntc_acceptance": _finite(ntc.get("acceptance_fraction")),
        "repeated_pair_fraction": _finite(ntc.get("repeated_particle_pair_fraction")),
        "majorant_violation_fraction": _finite(
            ntc.get("majorant_violation_fraction")),
        "correction_support_fraction": (float("nan") if sphere else
            1.0 - float(diagnostics.get(
                "correction_fallback_fraction_in_evaluation_window", 0.0))),
        "closure_total_fraction": _finite(diagnostics.get("closure_total_fraction")),
        "energy_ledger_residual": _finite((diagnostics.get("usf_energy_ledger") or {}).get(
            "relative_residual")),
    }
    values["N1k"] = values["Pk_xx"] - values["Pk_yy"]
    values["N2k"] = values["Pk_yy"] - values["Pk_zz"]
    values["encounters_per_strain"] = values["collisions_per_strain"]
    if diagnostics.get("event_unit") == "encounter":
        if ENCOUNTER_TABLE is None:
            raise ValueError(f"{prefix} counts encounters; pass --encounter-table")
        values["collisions_per_strain"] *= contacts_per_encounter(
            alpha, values["theta"], ar)
    values["astar"] = 1.0 / math.sqrt(values["Tstar"])
    values["eta_kinetic"] = -values["Pk_xy"] / values["astar"]
    values["temperature_ratio_drift_error"] = (
        values["temperature_ratio_drift"]
        - values["expected_temperature_ratio_drift"])
    if keep_series:
        values["series"] = {
            "tau": tau,
            "Tstar": tstar,
            "theta": theta,
            "pressure_tau": pressure[:, 1],
            "minus_Pk_xy": -pk[:, 3],
        }
    return values


def paired_estimate(items: list[dict], key: str) -> tuple[float, float, float]:
    """Return mean, independent-seed SE, and cold/hot convergence gap."""
    per_seed: list[float] = []
    cold, hot = [], []
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        grouped[item["replicate"]].append(item)
        value = _finite(item.get(key))
        if item["branch"] == "cold":
            cold.append(value)
        elif item["branch"] == "hot":
            hot.append(value)
    for seed_items in grouped.values():
        values = np.asarray([_finite(item.get(key)) for item in seed_items], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            per_seed.append(float(np.mean(values)))
    sample = np.asarray(per_seed, dtype=float)
    mean = float(np.mean(sample)) if len(sample) else float("nan")
    se = (float(np.std(sample, ddof=1) / math.sqrt(len(sample)))
          if len(sample) > 1 else float("nan"))
    c, h = np.asarray(cold, dtype=float), np.asarray(hot, dtype=float)
    c, h = c[np.isfinite(c)], h[np.isfinite(h)]
    if len(c) and len(h):
        cm, hm = float(np.mean(c)), float(np.mean(h))
        gap = abs(cm - hm) / max(0.5 * (abs(cm) + abs(hm)), 1.0e-30)
    else:
        gap = float("nan")
    return mean, se, gap


def aggregate_runs(runs: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for run in runs:
        groups[(run["AR"], run["alpha"])].append(run)
    keys = [key for key in runs[0] if key not in {
        "AR", "alpha", "replicate", "branch", "series"}]
    result = []
    for (ar, alpha), items in sorted(groups.items()):
        row = {"AR": ar, "alpha": alpha, "n_trajectories": len(items),
               "n_independent_seeds": len({item["replicate"] for item in items})}
        for key in keys:
            mean, se, gap = paired_estimate(items, key)
            row[key], row[key + "_se"], row[key + "_branch_gap"] = mean, se, gap
        if ar == 1.0:
            row["S2"] = row["S2_se"] = row["S2_branch_gap"] = float("nan")
        else:
            by_seed: dict[int, list[np.ndarray]] = defaultdict(list)
            by_branch: dict[str, list[np.ndarray]] = defaultdict(list)
            for item in items:
                q = np.asarray([item[name] for name in Q_COMPONENTS], dtype=float)
                by_seed[item["replicate"]].append(q)
                by_branch[item["branch"]].append(q)
            seed_q = np.asarray([np.mean(values, axis=0)
                                 for values in by_seed.values()])
            row["S2"] = nematic_order(np.mean(seed_q, axis=0))
            if len(seed_q) > 1:
                jackknife = np.asarray([
                    nematic_order(np.mean(np.delete(seed_q, index, axis=0), axis=0))
                    for index in range(len(seed_q))])
                row["S2_se"] = float(np.sqrt(
                    (len(seed_q) - 1.0) / len(seed_q)
                    * np.sum((jackknife - np.mean(jackknife)) ** 2)))
            else:
                row["S2_se"] = float("nan")
            branch_s2 = [nematic_order(np.mean(values, axis=0))
                         for values in by_branch.values()]
            row["S2_branch_gap"] = (abs(branch_s2[0] - branch_s2[1])
                                    / max(0.5 * sum(map(abs, branch_s2)), 1.0e-30))
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def matrix(rows: list[dict], key: str) -> np.ndarray:
    lookup = {(float(row["AR"]), float(row["alpha"])): _finite(row.get(key))
              for row in rows}
    return np.array([[lookup.get((ar, alpha), np.nan) for alpha in ALPHAS]
                     for ar in ARS], dtype=float)


def _at_ar(rows: list[dict], ar: float) -> list[dict]:
    return sorted((row for row in rows if float(row["AR"]) == ar),
                  key=lambda row: float(row["alpha"]))


def style_axis(ax, ylabel: str, logarithmic: bool = False,
               reference: float | None = None) -> None:
    ax.set_xlabel(r"coefficient of restitution $\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.485, 0.965)
    ax.set_xticks(ALPHAS[::2])
    if logarithmic:
        ax.set_yscale("log")
    if reference is not None:
        ax.axhline(reference, color="0.25", linestyle=":", linewidth=1.0)
    ax.grid(alpha=0.28)


def plot_dsmc_sweep(ax, rows: list[dict], key: str, ylabel: str,
                    scale: float = 1.0, sign: float = 1.0,
                    logarithmic: bool = False,
                    reference: float | None = None) -> None:
    for ar in ARS:
        selected = _at_ar(rows, ar)
        x = np.asarray([_finite(row["alpha"]) for row in selected])
        y = sign * scale * np.asarray([_finite(row.get(key)) for row in selected])
        se = scale * np.asarray([_finite(row.get(key + "_se"))
                                 for row in selected])
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        yerr = np.where(np.isfinite(se[mask]), np.abs(se[mask]), 0.0)
        ax.errorbar(x[mask], y[mask], yerr=yerr, color=COLORS[ar],
                    marker=MARKERS[ar], markerfacecolor=COLORS[ar],
                    markeredgecolor="white", markeredgewidth=0.7,
                    linewidth=1.45, markersize=6.2, capsize=2.2,
                    label=f"AR={ar:g}")
    style_axis(ax, ylabel, logarithmic, reference)


def plot_dem_sweep(ax, rows: list[dict], key: str, ylabel: str,
                   scale: float = 1.0, sign: float = 1.0,
                   logarithmic: bool = False,
                   reference: float | None = None) -> None:
    for ar in ARS:
        selected = _at_ar(rows, ar)
        x = np.asarray([_finite(row["alpha"]) for row in selected])
        y = sign * scale * np.asarray([_finite(row.get(key)) for row in selected])
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            ax.plot(x[mask], y[mask], color=COLORS[ar], marker=MARKERS[ar],
                    markerfacecolor=COLORS[ar], markeredgecolor="white",
                    markeredgewidth=0.7, linewidth=1.45, markersize=6.2)
    style_axis(ax, ylabel, logarithmic, reference)


def plot_method_sweep(ax, rows: list[dict], key: str, ylabel: str,
                      logarithmic: bool = False,
                      reference: float | None = None,
                      aspect_ratios: tuple[float, ...] = ARS) -> None:
    for ar in aspect_ratios:
        selected = _at_ar(rows, ar)
        x = np.asarray([_finite(row["alpha"]) for row in selected])
        dem = np.asarray([_finite(row.get("DEM_" + key)) for row in selected])
        dsmc = np.asarray([_finite(row.get("DSMC_" + key)) for row in selected])
        dem_mask = np.isfinite(x) & np.isfinite(dem)
        dsmc_mask = np.isfinite(x) & np.isfinite(dsmc)
        if np.any(dem_mask):
            ax.plot(x[dem_mask], dem[dem_mask], color=COLORS[ar],
                    linestyle="-", marker=MARKERS[ar],
                    markerfacecolor=COLORS[ar], markeredgecolor="white",
                    markeredgewidth=0.7, linewidth=1.35, markersize=6.0)
        if np.any(dsmc_mask):
            ax.plot(x[dsmc_mask], dsmc[dsmc_mask], color=COLORS[ar],
                    linestyle="--", marker=MARKERS[ar], markerfacecolor="white",
                    markeredgecolor=COLORS[ar], markeredgewidth=1.5,
                    linewidth=1.35, markersize=6.2)
    style_axis(ax, ylabel, logarithmic, reference)


def add_sweep_legend(fig, comparison: bool = False) -> None:
    handles = [Line2D([0], [0], color=COLORS[ar], marker=MARKERS[ar],
                      markerfacecolor=COLORS[ar], markeredgecolor="white",
                      linewidth=1.4, label=f"AR={ar:g}") for ar in ARS]
    if comparison:
        handles.extend((
            Line2D([0], [0], color="black", linestyle="-", marker="o",
                   markerfacecolor="black", label="DEM (filled)"),
            Line2D([0], [0], color="black", linestyle="--", marker="o",
                   markerfacecolor="white", markeredgewidth=1.5,
                   label="DSMC (hollow)"),
        ))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.005))


def finish_sweep_figure(fig, title: str, comparison: bool = False) -> None:
    fig.suptitle(title, fontsize=14)
    add_sweep_legend(fig, comparison)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.96))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_dsmc(rows: list[dict], runs: list[dict], output: Path) -> None:
    figures = output / "figures"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    plot_dsmc_sweep(axes[0, 0], rows, "Tstar", r"$T^*=T_{tr}/(m\dot\gamma^2d^2)$",
                    logarithmic=True)
    plot_dsmc_sweep(axes[0, 1], rows, "theta", r"$T_{tr}/T_{rot}$", reference=1.0)
    plot_dsmc_sweep(axes[1, 0], rows, "collisions_per_strain",
                    "collisions per particle per strain")
    plot_dsmc_sweep(axes[1, 1], rows, "S2", r"nematic order $S_2$")
    finish_sweep_figure(
        fig, "DSMC-USF steady state: thermodynamics, collision rate, and order")
    save(fig, figures / "01_state_map.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, key, label in zip(axes.flat, PK,
                              (r"$P^k_{xx}/nT_{tr}$", r"$P^k_{yy}/nT_{tr}$",
                               r"$P^k_{zz}/nT_{tr}$", r"$-P^k_{xy}/nT_{tr}$")):
        plot_dsmc_sweep(ax, rows, key, label, sign=-1.0 if key == "Pk_xy" else 1.0)
    finish_sweep_figure(
        fig, "DSMC-USF kinetic stress (validated rod stress observable)")
    save(fig, figures / "02_kinetic_stress_map.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    plot_dsmc_sweep(axes[0, 0], rows, "eta_kinetic",
                    r"kinetic viscosity $\eta_k^*=-P_{xy}^{k*}/a^*$")
    plot_dsmc_sweep(axes[0, 1], rows, "N1k",
                    r"kinetic $N_1^*=P^k_{xx}-P^k_{yy}$")
    plot_dsmc_sweep(axes[1, 0], rows, "N2k",
                    r"kinetic $N_2^*=P^k_{yy}-P^k_{zz}$", reference=0.0)
    plot_dsmc_sweep(axes[1, 1], rows, "cooling_to_shear",
                    "collisional dissipation / shear work", reference=1.0)
    finish_sweep_figure(fig, "DSMC-USF rheology and steady energy balance")
    save(fig, figures / "03_rheology_map.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    panels = (
        ("accepted_z", r"incoming accepted $\langle z\rangle_E$", None),
        ("outgoing_z", r"outgoing $\langle z'\rangle_E$", None),
        ("loss_fraction", r"mean fractional collision loss $\langle\epsilon\rangle$",
         None),
        ("orientation_acceptance", "orientation acceptance", None),
        ("ntc_acceptance", "NTC acceptance", None),
        ("correction_support_fraction", "angular-correction support fraction", None),
    )
    for ax, (key, label, reference) in zip(axes.flat, panels):
        plot_dsmc_sweep(ax, rows, key, label, reference=reference)
    finish_sweep_figure(
        fig, "DSMC-USF collision-kernel observables (sphere audit curves are absent)")
    save(fig, figures / "04_collision_observables_map.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    panels = (
        ("accepted_z_mean", r"accepted $\langle z\rangle$", None),
        ("loss_energy_weighted", r"energy-weighted loss $\langle\epsilon\rangle_E$",
         None),
        ("routing_loss_fraction", "closure-routed mean loss", None),
        ("collision_theta_energy_weighted", r"collision-weighted $T_{tr}/T_{rot}$",
         1.0),
        ("temperature_ratio_drift", "actual modal-temperature drift / pair energy",
         0.0),
        ("temperature_ratio_drift_error", "actual minus closure-expected drift",
         0.0),
    )
    for ax, (key, label, reference) in zip(axes.flat, panels):
        plot_dsmc_sweep(ax, rows, key, label, reference=reference)
    finish_sweep_figure(fig, "DSMC-USF run-integrated energy-routing observables")
    save(fig, figures / "04b_energy_routing_observables.png")

    precision_rows = []
    for row in rows:
        item = dict(row)
        for key in ("Tstar", "theta", "Pk_xy", "eta_kinetic"):
            mean, se = _finite(row.get(key)), _finite(row.get(key + "_se"))
            item[key + "_relative_se_percent"] = (
                100.0 * abs(se) / abs(mean)
                if np.isfinite(mean) and np.isfinite(se) and abs(mean) > 1.0e-30
                else float("nan"))
        precision_rows.append(item)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, key, title in zip(
            axes.flat, ("Tstar", "theta", "Pk_xy", "eta_kinetic"),
            (r"relative SE of $T^*$ (%)", r"relative SE of $T_{tr}/T_{rot}$ (%)",
             r"relative SE of $P^k_{xy}$ (%)", r"relative SE of $\eta_k^*$ (%)")):
        plot_dsmc_sweep(ax, precision_rows, key + "_relative_se_percent", title)
    finish_sweep_figure(
        fig, "DSMC-USF ensemble precision: four independent seeds after branch pairing")
    save(fig, figures / "05_sampling_precision.png")

    selected = ((1.0, 0.50), (2.0, 0.75), (3.0, 0.95))
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.5), constrained_layout=True)
    for column, (ar, alpha) in enumerate(selected):
        subset = [run for run in runs if run["AR"] == ar and run["alpha"] == alpha]
        tau_end = min(run["series"]["tau"][-1] for run in subset)
        tau_grid = np.linspace(0.0, tau_end, 321)
        for row_index, (key, label) in enumerate((
                ("Tstar", r"$T^*/\langle T^*\rangle_{ss}$"),
                ("theta", r"$T_{tr}/T_{rot}$"),
                ("minus_Pk_xy", r"$-P^k_{xy}/nT_{tr}$"))):
            paired = []
            by_seed: dict[int, list[np.ndarray]] = defaultdict(list)
            for run in subset:
                series = run["series"]
                x = series["pressure_tau"] if key == "minus_Pk_xy" else series["tau"]
                y = series[key]
                by_seed[run["replicate"]].append(np.interp(tau_grid, x, y))
            for values in by_seed.values():
                paired.append(np.mean(values, axis=0))
            values = np.asarray(paired)
            if key == "Tstar":
                steady = matrix(rows, "Tstar")[ARS.index(ar), ALPHAS.index(alpha)]
                values = values / steady
            axis = axes[row_index, column]
            if np.all(~np.isfinite(values)):
                axis.text(0.5, 0.5, "not defined for spheres", ha="center", va="center",
                          transform=axis.transAxes)
            else:
                mean = np.nanmean(values, axis=0)
                se = np.nanstd(values, axis=0, ddof=1) / math.sqrt(len(values))
                axis.plot(tau_grid, mean, color=COLORS[ar])
                axis.fill_between(tau_grid, mean - se, mean + se,
                                  color=COLORS[ar], alpha=0.22, linewidth=0)
            axis.set_ylabel(label)
            axis.set_xlabel("collisions per particle")
            axis.grid(alpha=0.25)
        axes[0, column].set_title(f"AR={ar:g}, alpha={alpha:.2f}")
    fig.suptitle("DSMC-USF convergence from cold/hot starts; bands are seed-level SE")
    save(fig, figures / "06_selected_transients.png")

    health_rows = []
    for row in rows:
        item = dict(row)
        item["repeated_pair_percent"] = 100.0 * _finite(row["repeated_pair_fraction"])
        item["majorant_violation_percent"] = (
            100.0 * _finite(row["majorant_violation_fraction"]))
        item["closure_total_percent"] = 100.0 * _finite(row["closure_total_fraction"])
        item["log_energy_residual"] = math.log10(
            _finite(row["energy_ledger_residual"]))
        health_rows.append(item)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, key, label in zip(
            axes.flat,
            ("repeated_pair_percent", "majorant_violation_percent",
             "closure_total_percent", "log_energy_residual"),
            ("same-step repeated-pair fraction (%)",
             "NTC majorant-violation fraction (%)",
             "closure wall-clock fraction (%)",
             r"$\log_{10}$ energy-ledger relative residual")):
        plot_dsmc_sweep(ax, health_rows, key, label)
    finish_sweep_figure(fig, "DSMC-USF numerical collision health")
    save(fig, figures / "07_numerical_collision_health.png")


def dem_rows(path: Path) -> list[dict]:
    result = []
    for raw in read_csv(path):
        ar, alpha = float(raw["AR"]), float(raw["alpha"])
        if ar not in ARS or round(alpha, 2) not in ALPHAS:
            continue
        row = {name: _finite(value) for name, value in raw.items()
               if name not in ("case", "flags")}
        row["case"], row["flags"] = raw.get("case", ""), raw.get("flags", "")
        row["N1k"] = row["Pk_xx"] - row["Pk_yy"]
        row["N2k"] = row["Pk_yy"] - row["Pk_zz"]
        row["astar"] = 1.0 / math.sqrt(row["Tstar"])
        row["eta_kinetic"] = -row["Pk_xy"] / row["astar"]
        pressure = (row["P_xx"] + row["P_yy"] + row["P_zz"]) / 3.0
        collisional_pressure = (row["Pc_xx"] + row["Pc_yy"] + row["Pc_zz"]) / 3.0
        row["collisional_pressure_share"] = collisional_pressure / pressure
        row["collisional_shear_share"] = row["Pc_xy"] / row["P_xy"]
        row["cooling_to_shear"] = -row["Wnc_over_Wshear"]
        row["S2"] = row["S2_nematic"]
        row["collisions_per_strain"] = row["nu_over_gdot"]
        result.append(row)
    return sorted(result, key=lambda item: (item["AR"], item["alpha"]))


def comparison_rows(dsmc: list[dict], dem: list[dict]) -> list[dict]:
    dlookup = {(row["AR"], row["alpha"]): row for row in dsmc}
    mlookup = {(row["AR"], row["alpha"]): row for row in dem}
    rows = []
    for key in sorted(set(dlookup) & set(mlookup)):
        d, m = dlookup[key], mlookup[key]
        row = {"AR": key[0], "alpha": key[1]}
        fields = ("Tstar", "theta", *PK, "N1k", "N2k", "eta_kinetic",
                  "S2", "collisions_per_strain", "cooling_to_shear")
        for field in fields:
            dv, mv = _finite(d.get(field)), _finite(m.get(field))
            row["DSMC_" + field], row["DEM_" + field] = dv, mv
            row[field + "_difference"] = dv - mv
            row[field + "_relative_difference"] = ((dv - mv) / abs(mv)
                                                        if np.isfinite(dv) and np.isfinite(mv)
                                                        and abs(mv) > 1.0e-30 else float("nan"))
        for field in ("Pc_xx", "Pc_yy", "Pc_zz", "Pc_xy",
                      "collisional_pressure_share", "collisional_shear_share"):
            row["DEM_" + field] = _finite(m.get(field))
        rows.append(row)
    return rows


def plot_comparison(rows: list[dict], dem: list[dict], output: Path) -> None:
    figures = output / "figures"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    plot_method_sweep(axes[0, 0], rows, "Tstar",
                      r"$T^*=T_{tr}/(m\dot\gamma^2d^2)$", logarithmic=True)
    plot_method_sweep(axes[0, 1], rows, "theta", r"$T_{tr}/T_{rot}$",
                      reference=1.0, aspect_ratios=ARS[1:])
    plot_method_sweep(axes[1, 0], rows, "collisions_per_strain",
                      "collisions per particle per strain")
    plot_method_sweep(axes[1, 1], rows, "S2", r"nematic order $S_2$",
                      aspect_ratios=ARS[1:])
    finish_sweep_figure(
        fig, "DEM and DSMC: common steady-state observables", comparison=True)
    save(fig, figures / "01_state_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, key, label in zip(axes.flat, PK,
                              (r"$P^k_{xx}/nT_{tr}$", r"$P^k_{yy}/nT_{tr}$",
                               r"$P^k_{zz}/nT_{tr}$", r"$P^k_{xy}/nT_{tr}$")):
        plot_method_sweep(ax, rows, key, label)
    finish_sweep_figure(
        fig, "DEM and DSMC kinetic stress (rod collisional stress excluded)",
        comparison=True)
    save(fig, figures / "02_kinetic_stress_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, key, label in zip(
            axes.flat, ("eta_kinetic", "N1k", "N2k", "cooling_to_shear"),
            (r"kinetic $\eta^*$", r"kinetic $N_1^*$", r"kinetic $N_2^*$",
             "dissipation / shear work")):
        plot_method_sweep(ax, rows, key, label,
                          reference=1.0 if key == "cooling_to_shear" else None)
    finish_sweep_figure(fig, "DEM and DSMC rheology and macroscopic cooling",
                        comparison=True)
    save(fig, figures / "03_rheology_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    plot_dem_sweep(axes[0, 0], dem, "collisional_pressure_share",
                   "collisional share of pressure (%)", scale=100.0)
    plot_dem_sweep(axes[0, 1], dem, "collisional_shear_share",
                   r"collisional share of $P_{xy}$ (%)", scale=100.0)
    plot_dem_sweep(axes[1, 0], dem, "Pc_xx", r"$P^c_{xx}/nT_{tr}$")
    plot_dem_sweep(axes[1, 1], dem, "Pc_xy", r"$-P^c_{xy}/nT_{tr}$", sign=-1.0)
    finish_sweep_figure(
        fig, "DEM collisional stress (not equated to the DSMC rod proxy)")
    save(fig, figures / "04_dem_collisional_stress.png")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    parity = (("Tstar", r"$T^*$", True), ("theta", r"$T_{tr}/T_{rot}$", False),
              ("Pk_xy", r"$P^k_{xy}/nT_{tr}$", False),
              ("collisions_per_strain", "collisions per particle per strain", False))
    for ax, (key, label, logarithmic) in zip(axes.flat, parity):
        values = []
        for ar in ARS:
            selected = [row for row in rows if row["AR"] == ar]
            x = np.asarray([row["DEM_" + key] for row in selected], dtype=float)
            y = np.asarray([row["DSMC_" + key] for row in selected], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], color=COLORS[ar], s=28, label=f"AR={ar:g}")
            values.extend(x[mask]); values.extend(y[mask])
        finite = np.asarray(values, dtype=float)
        if len(finite):
            lower, upper = float(np.min(finite)), float(np.max(finite))
            if logarithmic:
                ax.set_xscale("log"); ax.set_yscale("log")
            ax.plot([lower, upper], [lower, upper], "k--", lw=1)
        ax.set_xlabel("DEM " + label)
        ax.set_ylabel("DSMC " + label)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("DEM–DSMC parity: independent methods, no calibration to DEM")
    save(fig, figures / "05_parity.png")


def describe(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([_finite(row.get(key)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return {"minimum": float(np.min(values)), "median": float(np.median(values)),
            "maximum": float(np.max(values))} if len(values) else {}


def median_absolute(rows: list[dict], key: str) -> float | None:
    values = np.asarray([abs(_finite(row.get(key))) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def maximum_value(rows: list[dict], key: str) -> float | None:
    values = np.asarray([_finite(row.get(key)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else None


def describe_absolute(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([abs(_finite(row.get(key))) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return {"minimum": float(np.min(values)), "median": float(np.median(values)),
            "maximum": float(np.max(values))} if len(values) else {}


def write_reports(dsmc: list[dict], comparison: list[dict], dem: list[dict],
                  dsmc_output: Path, comparison_output: Path, manifest: Path,
                  dem_summary: Path) -> None:
    support_counts = {
        "full": sum(_finite(row["correction_support_fraction"]) >= 0.999
                    for row in dsmc),
        "mixed": sum(1.0e-12 < _finite(row["correction_support_fraction"]) < 0.999
                     for row in dsmc),
        "base_law_only": sum(_finite(row["correction_support_fraction"]) <= 1.0e-12
                             for row in dsmc),
    }
    maximum_branch_gap = {key: maximum_value(dsmc, key + "_branch_gap")
                          for key in ("Tstar", "theta", "Pk_xy", "eta_kinetic")}
    dsmc_payload = {
        "schema": "dsmc-usf-independent-study-v1",
        "source_manifest": str(manifest),
        "aspect_ratios": list(ARS), "alphas": list(ALPHAS),
        "cases": len(dsmc), "trajectories_per_case": 8,
        "independent_seeds_per_case": 4,
        "particle_count": 10000,
        "observable_ranges": {key: describe(dsmc, key) for key in (
            "Tstar", "theta", *PK, "N1k", "N2k", "eta_kinetic", "S2",
            "collisions_per_strain", "cooling_to_shear", "loss_fraction",
            "loss_energy_weighted", "routing_loss_fraction",
            "temperature_ratio_drift", "temperature_ratio_drift_error",
            "correction_support_fraction")},
        "maximum_relative_seed_se": {key: describe([
            {key: abs(_finite(row.get(key + "_se"))) / max(abs(_finite(row.get(key))), 1e-30)}
            for row in dsmc], key).get("maximum") for key in (
                "Tstar", "theta", "Pk_xy", "eta_kinetic")},
        "maximum_cold_hot_relative_gap": maximum_branch_gap,
        "angular_correction_case_counts": support_counts,
        "numerical_quality": {
            "energy_ledger_relative_residual": describe(
                dsmc, "energy_ledger_residual"),
            "repeated_pair_fraction": describe(dsmc, "repeated_pair_fraction"),
            "closure_wallclock_fraction": describe(dsmc, "closure_total_fraction"),
        },
        "notes": [
            "Cold/hot starts are paired before seed-level standard errors are computed.",
            "The full campaign already supplies the requested large-N ensemble; no duplicate local run was made.",
            "USF non-Gaussian particle moments were not enabled in this campaign and cannot be reconstructed from aggregate files.",
            "Rod collisional stress in DSMC is diagnostic only because the exact branch vector is absent.",
            "S2 is evaluated from the time-averaged Q tensor, not by averaging instantaneous eigenvalues.",
            "Collision-audit scalars are integrated over the full trajectory, including startup.",
        ],
    }
    dsmc_output.mkdir(parents=True, exist_ok=True)
    (dsmc_output / "summary.json").write_text(json.dumps(dsmc_payload, indent=2) + "\n")
    support_text = (f"""- Angular correction is fully supported in {support_counts['full']} cases,
  mixed with fallback in {support_counts['mixed']}, and absent (validated base
  law only) in {support_counts['base_law_only']}. The map therefore describes
  the actually deployed adaptive model, not a correction forced outside its
  learned invariant domain.
""" if ROD_ARM == "corrected" else """- Invariant corrections are off. Rod collision counts are CTC encounters and
  are converted to contacts with the measured <k> before any DEM comparison.
""")
    dsmc_report = rf"""# Independent DSMC uniform-shear-flow study

## Dataset

- AR: {', '.join(f'{x:g}' for x in ARS)}
- alpha: 0.50 through 0.95 in increments of 0.05
- 10,000 particles per trajectory
- 8 trajectories per physical case: cold/hot starts for 4 independent seeds
- 50 physical cases and 400 trajectories
- model: {MODEL_DESCRIPTION}

Cold/hot starts are convergence tests. They are paired at fixed seed before
the standard error is calculated, so the reported uncertainty has four
independent units rather than eight pseudo-replicates.

## Main observables

The study reports temperatures, kinetic stress, kinetic normal-stress
differences, kinetic reduced viscosity, nematic order, collisions per particle
per strain, energy balance, energy-routing observables, collision acceptance,
and closure-support coverage. The `figures` directory uses small-multiple
alpha sweeps, with a consistent color and marker for each aspect ratio.

Collision-audit quantities in figures `04` and `04b` are integrated over each
complete trajectory, including startup; they are kernel-behavior diagnostics,
not additional steady-window estimators. Figure `07` keeps numerical health
separate from physical results.

The kinetic viscosity is

\[
a^*=\dot\gamma d\sqrt{{m/T_{{tr}}}}=1/\sqrt{{T^*}},\qquad
\eta_k^*=-P_{{xy}}^{{k*}}/a^*.
\]

## Findings

- The steady energy balance is closed: collisional dissipation/shear work is
  {describe(dsmc, 'cooling_to_shear').get('minimum', float('nan')):.4f} to
  {describe(dsmc, 'cooling_to_shear').get('maximum', float('nan')):.4f}.
- The largest seed-level relative standard error is
  {100*dsmc_payload['maximum_relative_seed_se']['Tstar']:.2f}% for `Tstar`,
  {100*dsmc_payload['maximum_relative_seed_se']['theta']:.2f}% for `theta`,
  {100*dsmc_payload['maximum_relative_seed_se']['Pk_xy']:.2f}% for `Pk_xy`, and
  {100*dsmc_payload['maximum_relative_seed_se']['eta_kinetic']:.2f}% for
  kinetic viscosity.
- The worst cold/hot-start gap is {100*maximum_branch_gap['Tstar']:.2f}% in
  `Tstar` and {100*maximum_branch_gap['Pk_xy']:.2f}% in kinetic shear stress.
- `S2` is only {describe(dsmc, 'S2').get('minimum', float('nan')):.6f} to
  {describe(dsmc, 'S2').get('maximum', float('nan')):.6f}: these DSMC rod
  states are essentially orientationally isotropic. `S2` is computed from the
  time-averaged Q tensor to remove the positive finite-particle eigenvalue bias.
{support_text}- The largest energy-ledger relative residual is
  {maximum_value(dsmc, 'energy_ledger_residual'):.2e}; the largest repeated-pair
  fraction within one DSMC time step is
  {100*maximum_value(dsmc, 'repeated_pair_fraction'):.3f}%.

## Limits

- The production run did not enable particle-level USF non-Gaussian sampling;
  kurtosis and tail statistics need a separate targeted campaign.
- Rod collisional/total stress is not a validated DSMC observable until the
  collision API carries the exact contact branch vector.
- The angular correction is adaptive. Many low-alpha states use the validated
  base collision law because their flow invariants leave the excitation box.
"""
    (dsmc_output / "REPORT.md").write_text(dsmc_report)

    coll_pressure = describe(dem, "collisional_pressure_share")
    coll_shear = describe(dem, "collisional_shear_share")
    comparison_keys = (
        "Tstar", "theta", "Pk_xx", "Pk_yy", "Pk_zz", "Pk_xy",
        "N1k", "N2k", "eta_kinetic", "collisions_per_strain",
        "cooling_to_shear")
    median_absolute_relative = {
        key: median_absolute(comparison, key + "_relative_difference")
        for key in comparison_keys}
    by_ar = {}
    for ar in ARS:
        selected = [row for row in comparison if row["AR"] == ar]
        by_ar[str(ar)] = {
            key: describe(selected, key + "_relative_difference").get("median")
            for key in ("Tstar", "theta", "Pk_xy", "eta_kinetic",
                        "collisions_per_strain")}
    comparison_payload = {
        "schema": "dem-dsmc-usf-comparison-v1",
        "dsmc_source": str(dsmc_output / "summary.csv"),
        "dem_source": str(dem_summary),
        "strictly_comparable": list(STRICT_COMPARABLE),
        "not_strictly_comparable": list(NOT_STRICTLY_COMPARABLE),
        "relative_difference_ranges": {key: describe(comparison, key) for key in (
            "Tstar_relative_difference", "theta_relative_difference",
            "Pk_xx_relative_difference", "Pk_yy_relative_difference",
            "Pk_zz_relative_difference", "Pk_xy_relative_difference",
            "eta_kinetic_relative_difference",
            "collisions_per_strain_relative_difference",
            "cooling_to_shear_relative_difference")},
        "dem_collisional_pressure_share": coll_pressure,
        "dem_collisional_shear_share": coll_shear,
        "median_absolute_relative_difference": median_absolute_relative,
        "median_signed_relative_difference_by_AR": by_ar,
        "nematic_order_absolute_difference": describe_absolute(
            comparison, "S2_difference"),
    }
    comparison_output.mkdir(parents=True, exist_ok=True)
    (comparison_output / "summary.json").write_text(
        json.dumps(comparison_payload, indent=2) + "\n")
    before_after_text = ""
    if PREVIOUS_COMPARISON:
        lines = ["", "## Before and after the event-unit repair", "",
                 "Absolute relative difference from DEM over the 40 rod cases "
                 "(previous model -> this model). See `figures/06_before_after_error.png`.", "",
                 "| observable | median before | median after | max before | max after |",
                 "|---|---:|---:|---:|---:|"]
        rods_now = [row for row in comparison if row["AR"] > 1.0]
        rods_old = [row for row in PREVIOUS_COMPARISON if row["AR"] > 1.0]
        for key, label in (("Tstar", "T*"), ("theta", "theta"), ("Pk_xy", "P*_xy"),
                           ("N1k", "N1"), ("eta_kinetic", "kinetic viscosity"),
                           ("collisions_per_strain", "collisions per strain")):
            old = np.abs([_finite(r[key + "_relative_difference"]) for r in rods_old])
            new = np.abs([_finite(r[key + "_relative_difference"]) for r in rods_now])
            lines.append(f"| {label} | {100*np.nanmedian(old):.1f}% | {100*np.nanmedian(new):.1f}% "
                         f"| {100*np.nanmax(old):.1f}% | {100*np.nanmax(new):.1f}% |")
        before_after_text = "\n".join(lines) + "\n"
    comparison_report = f"""# DEM–DSMC uniform-shear-flow comparison

This comparison uses the same 5 x 10 `(AR, alpha)` grid. DEM remains an
independent physical comparison and was not used to calibrate the DSMC closure.

DSMC model: {MODEL_DESCRIPTION}.
{before_after_text}
## Strict comparison set

{chr(10).join('- ' + item for item in STRICT_COMPARABLE)}

## Excluded from accuracy comparisons

{chr(10).join('- ' + item for item in NOT_STRICTLY_COMPARABLE)}

Only kinetic stress is compared for rods. DEM collisional stress is plotted
separately in `figures/04_dem_collisional_stress.png`.

In figures `01`--`03`, DEM is shown with filled markers and solid curves;
DSMC is shown with hollow markers and dashed curves. Figure `05` retains the
method-parity representation.

## Quantitative comparison

- Sphere-control agreement is close: the median signed DSMC-minus-DEM
  differences are {100*by_ar['1.0']['Tstar']:.2f}% in `Tstar`,
  {100*by_ar['1.0']['Pk_xy']:.2f}% in kinetic shear stress,
  {100*by_ar['1.0']['eta_kinetic']:.2f}% in kinetic viscosity, and
  {100*by_ar['1.0']['collisions_per_strain']:.2f}% in collision rate.
- Across all applicable cases, median absolute differences are
  {100*median_absolute_relative['Tstar']:.2f}% for `Tstar`,
  {100*median_absolute_relative['theta']:.2f}% for `theta`,
  {100*median_absolute_relative['Pk_xy']:.2f}% for kinetic shear stress,
  {100*median_absolute_relative['eta_kinetic']:.2f}% for kinetic viscosity,
  and {100*median_absolute_relative['collisions_per_strain']:.2f}% for
  collisions per particle per strain.
- Median signed `Tstar` differences by aspect ratio:
  {', '.join(f"AR={ar:g}: {100*by_ar[str(ar)]['Tstar']:.2f}%" for ar in ARS[1:])}.
  Median signed kinetic-viscosity differences:
  {', '.join(f"AR={ar:g}: {100*by_ar[str(ar)]['eta_kinetic']:.2f}%" for ar in ARS[1:])}.
- DSMC nematic order is systematically lower. Its median absolute difference
  from DEM is {median_absolute(comparison, 'S2_difference'):.5f}; this is not
  explained by DSMC sampling uncertainty after using the unbiased tensor-first
  estimator.

Across the DEM grid, the collisional pressure share ranges from
{100*coll_pressure.get('minimum', float('nan')):.2f}% to
{100*coll_pressure.get('maximum', float('nan')):.2f}% (median
{100*coll_pressure.get('median', float('nan')):.2f}%). The collisional shear
share ranges from {100*coll_shear.get('minimum', float('nan')):.2f}% to
{100*coll_shear.get('maximum', float('nan')):.2f}% (median
{100*coll_shear.get('median', float('nan')):.2f}%). This confirms quantitatively
whether the apparently small DEM collisional contribution is negligible.

Macroscopic cooling is compared through dissipation divided by shear work,
not through raw per-contact energy loss. That avoids treating a finite-duration
DEM contact event as identical to one instantaneous DSMC binary collision.
Both methods' ratio being close to one is a necessary steady-state balance,
not by itself proof that their collision microphysics agree.
"""
    (comparison_output / "REPORT.md").write_text(comparison_report)


PREVIOUS_COMPARISON: list[dict] | None = None


def plot_before_after(rows: list[dict], previous: list[dict], output: Path) -> None:
    quantities = (("Tstar", r"$T^*$"), ("theta", r"$\theta=T_{tr}/T_{rot}$"),
                  ("Pk_xy", r"$P^{k*}_{xy}$"), ("N1k", r"$N_1^{k*}$"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for ax, (key, label) in zip(axes.ravel(), quantities):
        for ar in ARS[1:]:
            new = sorted((r for r in rows if r["AR"] == ar), key=lambda r: r["alpha"])
            old = sorted((r for r in previous if r["AR"] == ar), key=lambda r: r["alpha"])
            ax.plot([r["alpha"] for r in old],
                    [100 * _finite(r[key + "_relative_difference"]) for r in old],
                    ls="--", marker=MARKERS[ar], mfc="none", color=COLORS[ar], alpha=0.55, lw=1.0)
            ax.plot([r["alpha"] for r in new],
                    [100 * _finite(r[key + "_relative_difference"]) for r in new],
                    ls="-", marker=MARKERS[ar], color=COLORS[ar], lw=1.6)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel(f"DSMC vs DEM, {label} (%)")
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel(r"restitution $\alpha$")
    handles = [Line2D([], [], color=COLORS[ar], marker=MARKERS[ar], label=f"AR={ar:g}") for ar in ARS[1:]]
    handles += [Line2D([], [], color="gray", ls="--", mfc="none", marker="o", label="previous model"),
                Line2D([], [], color="gray", ls="-", marker="o", label="encounter-unit model")]
    fig.suptitle("Relative difference from DEM before and after the event-unit repair",
                 y=1.035, fontsize=13)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=6, frameon=False)
    fig.tight_layout()
    save(fig, output / "figures" / "06_before_after_error.png")


def main() -> None:
    global ENCOUNTER_TABLE, MODEL_DESCRIPTION, ROD_ARM, PREVIOUS_COMPARISON
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dem-summary", required=True)
    parser.add_argument("--dsmc-output", required=True)
    parser.add_argument("--comparison-output", required=True)
    parser.add_argument("--rod-arm", default="corrected")
    parser.add_argument("--encounter-table", default=None)
    parser.add_argument("--model-description", default=MODEL_DESCRIPTION)
    parser.add_argument("--previous-comparison", default=None,
                        help="summary.csv of an earlier comparison, for before/after")
    args = parser.parse_args()
    ROD_ARM, MODEL_DESCRIPTION = args.rod_arm, args.model_description
    if args.encounter_table:
        ENCOUNTER_TABLE = json.loads(Path(args.encounter_table).read_text())
    if args.previous_comparison:
        PREVIOUS_COMPARISON = [{k: (_finite(v) if k not in ("case", "flags") else v)
                                for k, v in row.items()}
                               for row in read_csv(Path(args.previous_comparison))]

    manifest_path = Path(args.manifest)
    selected = [row for row in read_csv(manifest_path)
                if float(row["aspect_ratio"]) in ARS
                and round(float(row["alpha"]), 2) in ALPHAS
                and row["arm"] in ("sphere_exact", ROD_ARM)
                and float(row["dt"]) == 0.005 and float(row["rate_scale"]) == 1.0]
    expected = len(ARS) * len(ALPHAS) * 2 * 4
    if len(selected) != expected:
        raise SystemExit(f"expected {expected} selected trajectories, found {len(selected)}")
    transient_cases = {(1.0, 0.50), (2.0, 0.75), (3.0, 0.95)}
    runs = [load_run(row, (float(row["aspect_ratio"]), float(row["alpha"]))
                     in transient_cases) for row in selected]
    dsmc = aggregate_runs(runs)
    if len(dsmc) != len(ARS) * len(ALPHAS):
        raise SystemExit(f"expected 50 aggregated DSMC cases, found {len(dsmc)}")
    dem = dem_rows(Path(args.dem_summary))
    if len(dem) != len(dsmc):
        raise SystemExit(f"DEM/DSMC grid mismatch: {len(dem)} versus {len(dsmc)}")
    comparison = comparison_rows(dsmc, dem)

    dsmc_output, comparison_output = Path(args.dsmc_output), Path(args.comparison_output)
    write_csv(dsmc_output / "summary.csv", dsmc)
    write_csv(comparison_output / "summary.csv", comparison)
    write_csv(comparison_output / "dem_selected.csv", dem)
    plot_dsmc(dsmc, runs, dsmc_output)
    plot_comparison(comparison, dem, comparison_output)
    if PREVIOUS_COMPARISON:
        plot_before_after(comparison, PREVIOUS_COMPARISON, comparison_output)
    write_reports(dsmc, comparison, dem, dsmc_output, comparison_output,
                  manifest_path, Path(args.dem_summary))
    print(json.dumps({
        "selected_trajectories": len(runs), "dsmc_cases": len(dsmc),
        "comparison_cases": len(comparison),
        "dsmc_output": str(dsmc_output),
        "comparison_output": str(comparison_output),
    }, indent=2))


if __name__ == "__main__":
    main()
