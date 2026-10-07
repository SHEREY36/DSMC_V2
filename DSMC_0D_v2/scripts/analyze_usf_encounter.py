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
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FIELDS = ["Tstar", "theta", "Pk_xx", "Pk_yy", "Pk_zz", "Pk_xy", "N1k", "N2k", "events_per_strain"]


def contacts_per_encounter(table: dict, alpha: float, theta: float, ar: float) -> float:
    rows = [r for r in table["contacts_per_encounter"] if abs(r["AR"] - ar) < 1e-9]
    if not rows:
        return float("nan")
    alphas = sorted({r["alpha"] for r in rows})
    values = []
    for a in alphas:
        sub = sorted((r for r in rows if r["alpha"] == a), key=lambda r: r["theta"])
        values.append(np.interp(np.log(theta), np.log([r["theta"] for r in sub]),
                                [r["k_mean"] for r in sub]))
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


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--benchmark", required=True,
                        help="DEM USF table, e.g. DSMC_0D_v2/reference/usf_benchmark_C1.csv "
                             "(usf_dem_and_legacy_v1.csv for Model R)")
    parser.add_argument("--encounter-table", default="DSMC_0D_v2/models/encounter_cross_section_v2.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prefixes = sorted(p[:-5] for p in glob.glob(f"{args.results}/*/*.json")
                      if not p.endswith((".summary.json",)))
    runs = [r for r in map(steady, prefixes) if r is not None]
    if not runs:
        raise SystemExit(f"no finished USF runs under {args.results}; nothing to analyze")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output.with_suffix(".runs.csv"), runs)
    table = json.loads(Path(args.encounter_table).read_text())
    # cold and hot trajectories of one seed are averaged first
    per_seed = defaultdict(list)
    for r in runs:
        per_seed[(r["arm"], r["AR"], r["alpha"], r["replicate"])].append(r)
    per_case = defaultdict(list)
    for (arm, ar, alpha, rep), items in per_seed.items():
        per_case[(arm, ar, alpha)].append({f: float(np.mean([i[f] for i in items])) for f in FIELDS})
    def number(text):
        try:
            return float(text)
        except (TypeError, ValueError):
            return float("nan")
    bench = {(float(r["AR"]), round(float(r["alpha"]), 4)): {k: number(v) for k, v in r.items()}
             for r in csv.DictReader(open(args.benchmark))}
    rows = []
    for (arm, ar, alpha), seeds in sorted(per_case.items()):
        row = {"arm": arm, "AR": ar, "alpha": alpha}
        for f in FIELDS:
            vals = np.array([s[f] for s in seeds])
            row[f] = float(vals.mean())
            row[f + "_sem"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
            row[f + "_count"] = len(vals)
        k = contacts_per_encounter(table, alpha, row["theta"], ar) if arm != "contact_legacy" else 1.0
        row["contacts_per_strain"] = row["events_per_strain"] * k
        ref = bench.get((ar, round(alpha, 4)))
        if ref:
            row.update({key: val for key, val in ref.items() if key not in ("AR", "alpha")})
            for q in ("Tstar", "theta", "Pk_xy", "N1k", "N2k", "Pk_yy", "Pk_zz"):
                row[f"{q}_vs_DEM"] = row[q] / ref[f"DEM_{q}"] - 1.0
                if f"legacy_{q}" in ref:          # the Model R benchmark also carries a legacy model
                    row[f"legacy_{q}_vs_DEM"] = ref[f"legacy_{q}"] / ref[f"DEM_{q}"] - 1.0
            row["collisions_vs_DEM"] = row["contacts_per_strain"] / ref["DEM_collisions_per_strain"] - 1.0
        rows.append(row)
    write_csv(output, rows)
    summary = {}
    for arm in sorted({r["arm"] for r in rows}):
        g = [r for r in rows if r["arm"] == arm and "Tstar_vs_DEM" in r]
        if not g:
            continue
        summary[arm] = {q: {"median_abs": float(np.nanmedian([abs(r[f"{q}_vs_DEM"]) for r in g])),
                            "max_abs": float(np.nanmax([abs(r[f"{q}_vs_DEM"]) for r in g]))}
                        for q in ("Tstar", "theta", "Pk_xy", "N1k")}
        summary[arm]["collisions"] = {"median_abs": float(np.median([abs(r["collisions_vs_DEM"]) for r in g]))}
        summary[arm]["cases"] = len(g)
    legacy = {(r["AR"], r["alpha"]): r for r in rows if "legacy_Tstar_vs_DEM" in r}.values()
    if legacy:
        summary["legacy_study_reference"] = {
            q: {"median_abs": float(np.nanmedian([abs(r[f"legacy_{q}_vs_DEM"]) for r in legacy])),
                "max_abs": float(np.nanmax([abs(r[f"legacy_{q}_vs_DEM"]) for r in legacy]))}
            for q in ("Tstar", "theta", "Pk_xy", "N1k")}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    try:
        make_figure(rows, output.with_suffix(".png"))
    except Exception as exc:  # the table is the deliverable; a plot failure must not hide it
        print("figure skipped:", exc)


def make_figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    quantities = (("Tstar", r"$T^*$"), ("theta", r"$\theta=T_{tr}/T_{rot}$"),
                  ("Pk_xy", r"$P^{k*}_{xy}$"), ("N1k", r"$N_1^{k*}$"))
    rows = [r for r in rows if "Tstar_vs_DEM" in r]
    ars = sorted({r["AR"] for r in rows})
    fig, axes = plt.subplots(len(quantities), len(ars), figsize=(3.2 * len(ars), 2.6 * len(quantities)),
                             sharex=True, squeeze=False)
    style = {"encounter_memory": ("#1b7837", "o", "encounter + angular memory"),
             "encounter": ("#762a83", "s", "encounter unit"),
             "contact_legacy": ("#999999", "^", "legacy (this code)")}
    for j, ar in enumerate(ars):
        sub = [r for r in rows if r["AR"] == ar]
        ref = sorted({r["alpha"]: r for r in sub}.values(), key=lambda r: r["alpha"])
        for i, (q, label) in enumerate(quantities):
            ax = axes[i, j]
            ax.plot([r["alpha"] for r in ref], [r[f"DEM_{q}"] for r in ref], "k-", lw=1.6, label="DEM")
            ax.plot([r["alpha"] for r in ref], [r[f"legacy_{q}"] for r in ref], color="#d6604d",
                    ls="--", lw=1.2, label="legacy study")
            for arm, (color, marker, name) in style.items():
                g = sorted((r for r in sub if r["arm"] == arm), key=lambda r: r["alpha"])
                if g:
                    ax.errorbar([r["alpha"] for r in g], [r[q] for r in g],
                                yerr=[r[q + "_sem"] for r in g], color=color, marker=marker,
                                ms=4, lw=0.8, label=name)
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
