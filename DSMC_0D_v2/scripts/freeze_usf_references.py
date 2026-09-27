#!/usr/bin/env python3
"""Freeze corrected DEM and independent sphere-DSMC USF references.

The LAMMPS repository is an independent comparison source.  This script copies
only reduced observables, uncertainties, and model-discrepancy diagnostics; it
does not fit or modify any closure parameter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


DEM_FIELDS = (
    "AR", "alpha", "windows", "Tstar", "Tstar_se", "theta", "theta_se",
    "a2_tr", "a2_tr_se", "a2_rot", "a2_rot_se",
    "Pk_xx", "Pk_xx_se", "Pk_yy", "Pk_yy_se",
    "Pk_zz", "Pk_zz_se", "Pk_xy", "Pk_xy_se",
    "Pc_xx", "Pc_xx_se", "Pc_yy", "Pc_yy_se",
    "Pc_zz", "Pc_zz_se", "Pc_xy", "Pc_xy_se",
    "P_xx", "P_xx_se", "P_yy", "P_yy_se",
    "P_zz", "P_zz_se", "P_xy", "P_xy_se",
    "S2_nematic", "S2_nematic_blockse", "drift_T", "drift_Pxy",
    "frac_binary", "frac_multibody", "frac_long", "frac_same_partner",
    "frac_rehit_0.01tau", "frac_rehit_0.1tau", "flag_windows", "flags",
)

SPHERE_FIELDS = (
    "alpha", "Tstar", "Pk_xx", "Pk_yy", "Pk_zz", "Pk_xy",
    "Tstar_uncertainty", "Pk_xx_uncertainty", "Pk_yy_uncertainty",
    "Pk_zz_uncertainty", "Pk_xy_uncertainty",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()


def freeze_dem(source: Path, output: Path) -> int:
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = sorted(set(DEM_FIELDS) - set(rows[0]))
    if missing:
        raise ValueError(f"DEM summary is missing columns: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEM_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row[name] for name in DEM_FIELDS} for row in rows)
    return len(rows)


def freeze_sphere(source: Path, output: Path) -> int:
    rows = []
    for line in source.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 11:
            raise ValueError(f"expected 11 sphere-reference columns, got {len(values)}")
        rows.append(dict(zip(SPHERE_FIELDS, values)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPHERE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps-root", required=True)
    parser.add_argument("--dem-output", default=(
        "DSMC_0D_v2/reference/usf_dem_fresh_v1.csv"))
    parser.add_argument("--sphere-output", default=(
        "DSMC_0D_v2/reference/usf_sphere_boltzmann_v1.csv"))
    parser.add_argument("--metadata-output", default=(
        "DSMC_0D_v2/reference/usf_reference_provenance_v1.json"))
    args = parser.parse_args()

    root = Path(args.lammps_root).resolve()
    dem_source = root / "runs/fresh_usf/summary.csv"
    sphere_source = root / "runs/fresh_usf/reference/dsmc_spheres_usf.txt"
    if not dem_source.is_file() or not sphere_source.is_file():
        raise SystemExit("LAMMPS fresh-USF reference files are missing")
    dem_output, sphere_output = Path(args.dem_output), Path(args.sphere_output)
    n_dem = freeze_dem(dem_source, dem_output)
    n_sphere = freeze_sphere(sphere_source, sphere_output)

    metadata = {
        "schema_version": "usf_reference_provenance_v1",
        "role": "independent_comparison_only_not_closure_calibration",
        "lammps_repository_revision": git_revision(root),
        "dem_source_relative_path": "runs/fresh_usf/summary.csv",
        "dem_source_sha256": sha256(dem_source),
        "dem_frozen_path": str(dem_output),
        "dem_frozen_sha256": sha256(dem_output),
        "dem_rows": n_dem,
        "sphere_source_relative_path": (
            "runs/fresh_usf/reference/dsmc_spheres_usf.txt"),
        "sphere_source_sha256": sha256(sphere_source),
        "sphere_frozen_path": str(sphere_output),
        "sphere_frozen_sha256": sha256(sphere_output),
        "sphere_rows": n_sphere,
        "stress_policy": {
            "primary_rod_comparison": "kinetic_stress",
            "rod_total_stress": "diagnostic_only_missing_exact_DSMC_branch_virial",
            "sphere_benchmark": "kinetic_stress_against_Boltzmann_DSMC",
        },
    }
    metadata_output = Path(args.metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
