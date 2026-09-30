#!/usr/bin/env python3
"""Steady-state analysis of the encounter-unit USF campaign against DEM.

Per trajectory: time averages over tau >= evaluation_start_tau of
  T* = T_tr/(m gdot^2 d^2),  theta = T_tr/T_rot,  P^k_ab/(n T_tr),  N1, N2,
  DSMC events per particle per unit strain.
Per (arm, AR, alpha): cold and hot trajectories of one seed are averaged
first, so the standard error has one degree of freedom per independent seed.

Encounter arms count one event per CTC encounter.  To compare the collision
rate with DEM contacts it is multiplied by <k>(alpha, theta, AR), the mean
contacts per encounter measured in the same CTC shards (encounter table).
DEM is an external benchmark here; nothing in the model was fitted to it.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def contacts_per_encounter(table: dict, alpha: float, theta: float, ar: float) -> float:
    rows = pd.DataFrame(table["contacts_per_encounter"])
    rows = rows[np.isclose(rows.AR, ar)]
    if rows.empty:
        return np.nan
    values = []
    alphas = np.sort(rows.alpha.unique())
    for a in alphas:
        sub = rows[np.isclose(rows.alpha, a)].sort_values("theta")
        values.append(np.interp(np.log(theta), np.log(sub.theta), sub.k_mean))
    return float(np.interp(alpha, alphas, values))


def steady(prefix: str) -> dict | None:
    meta = Path(prefix + ".json")
    if not meta.is_file():
        return None
    d = json.loads(meta.read_text())
    case = d["validation_case"]
    tr = np.loadtxt(prefix + ".txt"); pr = np.loadtxt(prefix + "_pressure.txt")
    start = float(case["evaluation_start_tau"]); gdot = float(case["shear_rate"])
    tau, ttr, trot, time = tr[:, 1], tr[:, 2], tr[:, 3], tr[:, 0]
    m = tau >= start
    n = float(d["number_density"])
    ttr_p = np.interp(pr[:, 1], tau, ttr); mp = pr[:, 1] >= start
    pk = {name: float(np.mean(pr[mp, j] / (n * ttr_p[mp])))
          for name, j in (("Pk_xx", 2), ("Pk_xy", 3), ("Pk_yy", 5), ("Pk_zz", 7))}
    mass = float(d.get("particle_mass", 1.0)); diameter = float(d.get("particle_diameter", 1.0))
    return dict(arm=case["arm"], AR=float(case["aspect_ratio"]), alpha=float(case["alpha"]),
                branch=case["initial_branch"], replicate=int(case["replicate"]),
                Tstar=float(np.mean(ttr[m])) / (mass * (gdot * diameter) ** 2),
                theta=float(np.mean(ttr[m] / trot[m])), **pk,
                N1k=pk["Pk_xx"] - pk["Pk_yy"], N2k=pk["Pk_yy"] - pk["Pk_zz"],
                events_per_strain=float((tau[m][-1] - tau[m][0]) / ((time[m][-1] - time[m][0]) * gdot)),
                event_unit=d.get("event_unit"), wall=float(d.get("runtime_seconds", np.nan)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--benchmark", default="DSMC_0D_v2/reference/usf_dem_and_legacy_v1.csv")
    parser.add_argument("--encounter-table", default="DSMC_0D_v2/models/encounter_cross_section_v2.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prefixes = sorted(p[:-5] for p in glob.glob(f"{args.results}/*/*.json"))
    runs = pd.DataFrame([r for r in map(steady, prefixes) if r is not None])
    runs.to_csv(Path(args.output).with_suffix(".runs.csv"), index=False)
    table = json.loads(Path(args.encounter_table).read_text())
    fields = ["Tstar", "theta", "Pk_xx", "Pk_yy", "Pk_zz", "Pk_xy", "N1k", "N2k", "events_per_strain"]
    seed = runs.groupby(["arm", "AR", "alpha", "replicate"])[fields].mean().reset_index()
    agg = seed.groupby(["arm", "AR", "alpha"])[fields].agg(["mean", "sem", "count"])
    agg.columns = [f"{a}_{b}" if b != "mean" else a for a, b in agg.columns]
    agg = agg.reset_index()
    agg["contacts_per_strain"] = [
        row.events_per_strain * (contacts_per_encounter(table, row.alpha, row.theta, row.AR)
                                 if row.arm != "contact_legacy" else 1.0)
        for row in agg.itertuples()]
    bench = pd.read_csv(args.benchmark)
    merged = agg.merge(bench, on=["AR", "alpha"], how="left")
    for q in ("Tstar", "theta", "Pk_xy", "N1k", "N2k", "Pk_yy", "Pk_zz"):
        merged[f"{q}_vs_DEM"] = merged[q] / merged[f"DEM_{q}"] - 1.0
        merged[f"legacy_{q}_vs_DEM"] = merged[f"legacy_{q}"] / merged[f"DEM_{q}"] - 1.0
    merged["collisions_vs_DEM"] = merged.contacts_per_strain / merged.DEM_collisions_per_strain - 1.0
    merged.to_csv(args.output, index=False)
    summary = {}
    for arm, g in merged.groupby("arm"):
        summary[arm] = {q: {"median_abs": float(np.nanmedian(np.abs(g[f"{q}_vs_DEM"]))),
                            "max_abs": float(np.nanmax(np.abs(g[f"{q}_vs_DEM"])))}
                        for q in ("Tstar", "theta", "Pk_xy", "N1k")}
        summary[arm]["collisions"] = {"median_abs": float(np.nanmedian(np.abs(g.collisions_vs_DEM)))}
        summary[arm]["cases"] = int(len(g))
    legacy = merged.drop_duplicates(["AR", "alpha"])
    summary["legacy_study_reference"] = {
        q: {"median_abs": float(np.nanmedian(np.abs(legacy[f"legacy_{q}_vs_DEM"]))),
            "max_abs": float(np.nanmax(np.abs(legacy[f"legacy_{q}_vs_DEM"])))}
        for q in ("Tstar", "theta", "Pk_xy", "N1k")}
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    try:
        make_figure(merged, Path(args.output).with_suffix(".png"))
    except Exception as exc:  # the table is the deliverable; a plot failure must not hide it
        print("figure failed:", exc)


def make_figure(merged: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    quantities = (("Tstar", r"$T^*$"), ("theta", r"$\theta=T_{tr}/T_{rot}$"),
                  ("Pk_xy", r"$P^{k*}_{xy}$"), ("N1k", r"$N_1^{k*}$"))
    ars = sorted(merged.AR.unique())
    fig, axes = plt.subplots(len(quantities), len(ars), figsize=(3.2 * len(ars), 2.6 * len(quantities)),
                             sharex=True, squeeze=False)
    style = {"encounter_memory": ("#1b7837", "o", "encounter + angular memory"),
             "encounter": ("#762a83", "s", "encounter unit"),
             "contact_legacy": ("#999999", "^", "legacy (this code)")}
    for j, ar in enumerate(ars):
        sub = merged[merged.AR == ar]
        ref = sub.drop_duplicates("alpha").sort_values("alpha")
        for i, (q, label) in enumerate(quantities):
            ax = axes[i, j]
            ax.plot(ref.alpha, ref[f"DEM_{q}"], "k-", lw=1.6, label="DEM")
            ax.plot(ref.alpha, ref[f"legacy_{q}"], color="#d6604d", ls="--", lw=1.2,
                    label="legacy study")
            for arm, (color, marker, name) in style.items():
                g = sub[sub.arm == arm].sort_values("alpha")
                if len(g):
                    ax.errorbar(g.alpha, g[q], yerr=g.get(f"{q}_sem"), color=color, marker=marker,
                                ms=4, lw=0.8, ls=":" if arm != "encounter_memory" else "-",
                                mfc="none" if arm != "encounter_memory" else color, label=name)
            if i == 0:
                ax.set_title(f"AR = {ar:g}")
            if j == 0:
                ax.set_ylabel(label)
            if q == "Tstar":
                ax.set_yscale("log")
    axes[-1, 0].set_xlabel(r"$\alpha$")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()
