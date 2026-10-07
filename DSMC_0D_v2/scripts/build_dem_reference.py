#!/usr/bin/env python3
"""Freeze the DEM reference tables for the DSMC comparisons (one contact model per table).

    python DSMC_0D_v2/scripts/build_dem_reference.py --lammps-root ../LAMMPS --tag C1 \
        --provenance ../ModelC_campaign/negishi_records/operations/<time>_production_start.json

Reads the reduced LAMMPS runs and writes, in DSMC_0D_v2/reference/:
  hcs_dem_<tag>.csv   one row per (AR, alpha): seed means of runs/fresh_hcs/analysis/summary.csv
                      (theta_star, cooling rate, alpha_eff, sigma_eff, e_c, a2) with the
                      standard error of the mean over seeds and the seed spread (sample std)
  usf_dem_<tag>.csv   one row per (AR, alpha): mean of the USF realizations s1, s2
                      (runs/fresh_usf/s*/summary.csv); *_se combines the block errors of the
                      realizations, *_seed_spread is their sample standard deviation
  usf_benchmark_<tag>.csv   the same USF data in the format of analyze_usf_encounter.py
                      --benchmark (DEM_Tstar, DEM_theta, DEM_Pk_*, DEM_N1k = Pk_xx - Pk_yy,
                      DEM_N2k = Pk_yy - Pk_zz, DEM_collisions_per_strain = nu/gdot, DEM_S2)
  dem_reference_provenance_<tag>.json   sources and their sha256, LAMMPS revision and binary,
                      contact model, frozen-file sha256

Every source row must be of the requested contact model (and, for HCS, have no positive
damping step); anything else is an error.  The tables are for independent comparison only,
never for calibrating the closure.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "..", "reference")

HCS_COLUMNS = ["theta_star", "gamma", "alpha_eff", "sigma_eff", "e_c", "e_tr", "a2_tr", "a2_rot",
               "frac_multibody", "energy_closure"]
USF_COLUMNS = ["Tstar", "theta", "a2_tr", "a2_rot",
               "Pk_xx", "Pk_yy", "Pk_zz", "Pk_xy", "Pc_xx", "Pc_yy", "Pc_zz", "Pc_xy",
               "P_xx", "P_yy", "P_zz", "P_xy", "N1", "N2", "nu_over_gdot", "S2_nematic"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mean_sd(values):
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def hcs(path, tag):
    groups = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["contact_model"] != tag:
            raise SystemExit(f"{path}: AR {r['AR']} alpha {r['alpha']} {r['seed']} is contact model "
                             f"{r['contact_model']}, not {tag}")
        if tag == "C1" and float(r["frac_dpos"]) != 0.0:
            raise SystemExit(f"{path}: positive damping work in AR {r['AR']} alpha {r['alpha']} {r['seed']}")
        groups[(float(r["AR"]), round(float(r["alpha"]), 4))].append(r)
    rows = []
    for (ar, a), runs in sorted(groups.items()):
        row = {"AR": ar, "alpha": a}
        th, sd = mean_sd([float(r["theta_star"]) for r in runs])
        row.update(theta_star=th, theta_star_seed_spread=sd, seeds=len(runs),
                   theta_star_se=sd / math.sqrt(len(runs)))
        for c in HCS_COLUMNS[1:]:
            m, s = mean_sd([float(r[c]) for r in runs])
            row[c], row[c + "_se"] = m, s / math.sqrt(len(runs))
        rows.append(row)
    fields = ["AR", "alpha", "theta_star", "theta_star_seed_spread", "seeds", "theta_star_se"] + \
        [x for c in HCS_COLUMNS[1:] for x in (c, c + "_se")]
    return rows, fields


def usf(paths, tag):
    groups = defaultdict(list)
    for path in paths:
        for r in csv.DictReader(open(path)):
            if r["contact_model"] != tag:
                raise SystemExit(f"{path}: {r['case']} is contact model {r['contact_model']}, not {tag}")
            groups[(float(r["AR"]), round(float(r["alpha"]), 4))].append(r)
    rows = []
    for (ar, a), runs in sorted(groups.items()):
        n = len(runs)
        row = {"AR": ar, "alpha": a, "seeds": n, "windows": sum(int(r["windows"]) for r in runs),
               "flags": " ".join(r["flags"] for r in runs)}
        for c in USF_COLUMNS:
            m, sd = mean_sd([float(r[c]) for r in runs])
            row[c], row[c + "_seed_spread"] = m, sd
            se_col = c + "_se" if c + "_se" in runs[0] else c + "_blockse" if c + "_blockse" in runs[0] else None
            row[c + "_se"] = math.sqrt(sum(float(r[se_col]) ** 2 for r in runs)) / n if se_col else float("nan")
        rows.append(row)
    fields = ["AR", "alpha", "seeds", "windows"] + \
        [x for c in USF_COLUMNS for x in (c, c + "_se", c + "_seed_spread")] + ["flags"]
    return rows, fields


def usf_benchmark(rows):
    """USF rows in the --benchmark format of analyze_usf_encounter.py (DEM columns only)."""
    out = []
    for r in rows:
        b = {"AR": r["AR"], "alpha": r["alpha"]}
        for name, src in (("Tstar", "Tstar"), ("theta", "theta"), ("Pk_xx", "Pk_xx"), ("Pk_yy", "Pk_yy"),
                          ("Pk_zz", "Pk_zz"), ("Pk_xy", "Pk_xy"), ("collisions_per_strain", "nu_over_gdot"),
                          ("S2", "S2_nematic")):
            b["DEM_" + name], b["DEM_" + name + "_se"] = r[src], r[src + "_se"]
        b["DEM_N1k"] = r["Pk_xx"] - r["Pk_yy"]
        b["DEM_N2k"] = r["Pk_yy"] - r["Pk_zz"]
        out.append(b)
    return out, list(out[0])


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{path}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lammps-root", required=True, help="MD_LAMMPS checkout holding runs/fresh_hcs, fresh_usf")
    ap.add_argument("--tag", required=True, help="contact model of the runs, e.g. C1")
    ap.add_argument("--provenance", required=True,
                    help="campaign start record (LAMMPS revision and lmp_mpi sha256 of the runs)")
    ap.add_argument("--usf-sets", nargs="+", default=["s1", "s2"])
    ap.add_argument("--out-dir", default=REFERENCE)
    a = ap.parse_args()
    root = os.path.abspath(a.lammps_root)
    hcs_src = os.path.join(root, "runs", "fresh_hcs", "analysis", "summary.csv")
    usf_src = [os.path.join(root, "runs", "fresh_usf", s, "summary.csv") for s in a.usf_sets]
    rec = json.load(open(a.provenance))
    os.makedirs(a.out_dir, exist_ok=True)
    out = {}
    for name, (rows, fields), sources in (("hcs", hcs(hcs_src, a.tag), [hcs_src]),
                                          ("usf", usf(usf_src, a.tag), usf_src)):
        path = os.path.join(a.out_dir, f"{name}_dem_{a.tag}.csv")
        write(path, rows, fields)
        out[name] = {"frozen_path": os.path.relpath(path, os.path.join(HERE, "..", "..")),
                     "frozen_sha256": sha256(path), "rows": len(rows),
                     "sources": [{"relative_path": os.path.relpath(s, root), "sha256": sha256(s)}
                                 for s in sources]}
        if name == "usf":
            bpath = os.path.join(a.out_dir, f"usf_benchmark_{a.tag}.csv")
            write(bpath, *usf_benchmark(rows))
            out[name]["benchmark_path"] = os.path.relpath(bpath, os.path.join(HERE, "..", ".."))
            out[name]["benchmark_sha256"] = sha256(bpath)
    prov = {"schema_version": "dem_reference_provenance_v2", "contact_model_id": a.tag,
            "damp_velocity": "contact" if a.tag == "C1" else "center",
            "role": "independent_comparison_only_not_closure_calibration",
            "lammps_repository_revision_at_campaign_start": rec["repos"]["MD_LAMMPS"]["head"],
            "lammps_binary_sha256": rec["binaries"]["lmp_mpi"],
            "campaign_record": os.path.basename(a.provenance), "campaign_tag": rec.get("tag"),
            "hcs": out["hcs"], "usf": out["usf"],
            "statistics": {"hcs": "mean over seeds; *_se = sample std / sqrt(seeds)",
                           "usf": "mean over realizations; *_se = sqrt(sum of block se^2) / n; "
                                  "*_seed_spread = sample std of the realizations"}}
    path = os.path.join(a.out_dir, f"dem_reference_provenance_{a.tag}.json")
    json.dump(prov, open(path, "w"), indent=2)
    print(f"{path}")


if __name__ == "__main__":
    main()
