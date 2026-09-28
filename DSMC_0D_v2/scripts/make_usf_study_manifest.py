#!/usr/bin/env python3
"""Create the numerical, cross-flow pilot, or full USF study manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


SHEAR_RATES = {
    0.50: 0.0800, 0.55: 0.0771, 0.60: 0.0739, 0.65: 0.0702,
    0.70: 0.0660, 0.75: 0.0611, 0.80: 0.0554, 0.85: 0.0487,
    0.90: 0.0403, 0.95: 0.0288,
}
SEEDS = (260927101, 260927211, 260927307, 260927419, 260927523, 260927641)
BRANCHES = (("cold", 0.5, 0.75), ("hot", 2.0, 1.25))
ARTIFACT_ARS = (1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0)
INTERPOLATION_CASES = tuple(
    (ar, alpha) for ar in (1.15, 1.275, 1.425, 1.75, 2.25, 2.75)
    for alpha in (0.65, 0.875))


def load_initialization_reference(path: Path) -> dict[float, dict[float, tuple[float, float]]]:
    result: dict[float, dict[float, tuple[float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ar, alpha = float(row["AR"]), float(row["alpha"])
            result.setdefault(alpha, {})[ar] = (
                float(row["Tstar"]), float(row["theta"]))
    return result


def initialization_guess(reference: dict, ar: float, alpha: float) -> tuple[float, float]:
    alphas = np.asarray(sorted(reference), dtype=float)
    tstar_by_alpha, theta_by_alpha = [], []
    for node_alpha in alphas:
        family = reference[float(node_alpha)]
        ars = np.asarray(sorted(family), dtype=float)
        tstar_by_alpha.append(np.interp(ar, ars, [family[x][0] for x in ars]))
        theta_by_alpha.append(np.interp(ar, ars, [family[x][1] for x in ars]))
    # DEM is used only to avoid a needlessly long transient.  Values outside
    # its rod grid use the nearest endpoint; both hot/cold branches must still
    # converge independently before a DSMC result is accepted.
    return (float(np.interp(alpha, alphas, tstar_by_alpha)),
            float(np.interp(alpha, alphas, theta_by_alpha)))


def shear_rate(alpha: float) -> float:
    nodes = np.asarray(sorted(SHEAR_RATES), dtype=float)
    return float(np.interp(alpha, nodes, [SHEAR_RATES[x] for x in nodes]))


def tau_end(alpha: float) -> float:
    if alpha <= 0.80:
        return 160.0
    if alpha <= 0.90:
        return 240.0
    return 360.0


def add_coordinate(rows: list[dict[str, object]], *, mode: str, role: str,
                   ar: float, alpha: float, arm: str, dt_values: tuple[float, ...],
                   rate_scales: tuple[float, ...], replicates: int,
                   particles: int, phi: float, results: Path,
                   reference: dict) -> None:
    sphere = arm == "sphere_exact"
    guess_tstar, guess_theta = initialization_guess(reference, ar, alpha)
    for dt in dt_values:
        for rate_scale in rate_scales:
            imposed_rate = shear_rate(alpha) * rate_scale
            for branch, temperature_scale, theta_scale in BRANCHES:
                initial_ttr = guess_tstar * imposed_rate**2 * temperature_scale
                initial_theta = 1.0 if sphere else guess_theta * theta_scale
                initial_trot = 0.0 if sphere else initial_ttr / initial_theta
                for replicate, seed in enumerate(SEEDS[:replicates]):
                    prefix = results / (
                        f"AR_{ar:.3f}_alpha_{alpha:.3f}_{arm}_{branch}_"
                        f"dt_{dt:.4f}_rate_{rate_scale:.2f}_rep_{replicate:02d}")
                    rows.append({
                        "task_id": len(rows), "mode": mode,
                        "coordinate_role": role, "arm": arm,
                        "sphere": str(sphere).lower(),
                        "alpha": f"{alpha:.3f}", "aspect_ratio": f"{ar:.3f}",
                        "shear_rate": f"{imposed_rate:.8g}",
                        "rate_scale": f"{rate_scale:.4g}", "dt": f"{dt:.8g}",
                        "initial_branch": branch,
                        "initial_ttr": f"{initial_ttr:.12g}",
                        "initial_trot": f"{initial_trot:.12g}",
                        "replicate": replicate, "seed": seed,
                        "particles": particles, "volume_fraction": f"{phi:.8g}",
                        "tau_end": f"{tau_end(alpha):.1f}",
                        "evaluation_start_tau": f"{0.60 * tau_end(alpha):.1f}",
                        "state_update_cpp": "0.05",
                        "output_prefix": str(prefix),
                    })


def campaign_rows(mode: str, results: Path, particles: int, phi: float,
                  replicates: int, reference: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if mode == "numerics":
        for alpha in (0.50, 0.80, 0.95):
            add_coordinate(rows, mode=mode, role="sphere_control", ar=1.0,
                           alpha=alpha, arm="sphere_exact",
                           dt_values=(0.01, 0.005), rate_scales=(1.0,),
                           replicates=replicates, particles=particles, phi=phi,
                           results=results, reference=reference)
        for ar in (1.5, 2.0, 3.0):
            for alpha in (0.50, 0.80, 0.95):
                add_coordinate(rows, mode=mode, role="rod_numerics", ar=ar,
                               alpha=alpha, arm="corrected",
                               dt_values=(0.01, 0.005), rate_scales=(1.0,),
                               replicates=replicates, particles=particles, phi=phi,
                               results=results, reference=reference)
            add_coordinate(rows, mode=mode, role="rate_invariance", ar=ar,
                           alpha=0.80, arm="corrected", dt_values=(0.005,),
                           rate_scales=(0.70,), replicates=replicates,
                           particles=particles, phi=phi, results=results,
                           reference=reference)
    elif mode == "pilot":
        for alpha in (0.50, 0.80, 0.95):
            add_coordinate(rows, mode=mode, role="sphere_control", ar=1.0,
                           alpha=alpha, arm="sphere_exact", dt_values=(0.005,),
                           rate_scales=(1.0,), replicates=replicates,
                           particles=particles, phi=phi, results=results,
                           reference=reference)
        for ar in (1.5, 2.0, 3.0):
            for alpha in (0.50, 0.80, 0.95):
                for arm in ("uncorrected", "corrected"):
                    add_coordinate(rows, mode=mode, role="paired_rod_pilot", ar=ar,
                                   alpha=alpha, arm=arm, dt_values=(0.005,),
                                   rate_scales=(1.0,), replicates=replicates,
                                   particles=particles, phi=phi, results=results,
                                   reference=reference)
    elif mode == "full":
        for alpha in sorted(SHEAR_RATES):
            add_coordinate(rows, mode=mode, role="sphere_control", ar=1.0,
                           alpha=alpha, arm="sphere_exact", dt_values=(0.005,),
                           rate_scales=(1.0,), replicates=replicates,
                           particles=particles, phi=phi, results=results,
                           reference=reference)
        for ar in ARTIFACT_ARS:
            for alpha in sorted(SHEAR_RATES):
                add_coordinate(rows, mode=mode, role="artifact_node", ar=ar,
                               alpha=alpha, arm="corrected", dt_values=(0.005,),
                               rate_scales=(1.0,), replicates=replicates,
                               particles=particles, phi=phi, results=results,
                               reference=reference)
        for ar, alpha in INTERPOLATION_CASES:
            add_coordinate(rows, mode=mode, role="interpolation_holdout", ar=ar,
                           alpha=alpha, arm="corrected", dt_values=(0.005,),
                           rate_scales=(1.0,), replicates=replicates,
                           particles=particles, phi=phi, results=results,
                           reference=reference)
    elif mode == "response-validation":
        # Paired base/candidate dynamics on every direct-response coordinate.
        # No DEM value is a target or gate; its frozen table is used only by
        # add_coordinate to shorten the cold/hot transient.
        for alpha in (0.50, 0.80, 0.95):
            add_coordinate(
                rows, mode=mode, role="sphere_control", ar=1.0,
                alpha=alpha, arm="sphere_exact", dt_values=(0.005,),
                rate_scales=(1.0,), replicates=replicates,
                particles=particles, phi=phi, results=results,
                reference=reference)
        for ar in ARTIFACT_ARS:
            for alpha in (0.50, 0.80, 0.95):
                for arm in ("uncorrected", "corrected"):
                    add_coordinate(
                        rows, mode=mode, role="direct_response_validation",
                        ar=ar, alpha=alpha, arm=arm, dt_values=(0.005,),
                        rate_scales=(1.0,), replicates=replicates,
                        particles=particles, phi=phi, results=results,
                        reference=reference)
        # Untouched tensor-cell midpoints test interpolation across every AR
        # interval and both inelastic alpha intervals. They are validation
        # states only and never enter the direct-response fit.
        for ar, alpha in INTERPOLATION_CASES:
            for arm in ("uncorrected", "corrected"):
                add_coordinate(
                    rows, mode=mode, role="response_interpolation_holdout",
                    ar=ar, alpha=alpha, arm=arm, dt_values=(0.005,),
                    rate_scales=(1.0,), replicates=replicates,
                    particles=particles, phi=phi, results=results,
                    reference=reference)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("numerics", "pilot", "full",
                                            "response-validation"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--reference", default="DSMC_0D_v2/reference/usf_dem_fresh_v1.csv")
    parser.add_argument("--artifact", default=(
        "models/microscopic_closure_v2_angular_evidence/closure_v2.npz"))
    parser.add_argument("--particles", type=int, default=10_000)
    parser.add_argument("--volume-fraction", type=float, default=0.01)
    parser.add_argument("--replicates", type=int)
    args = parser.parse_args()
    if args.particles < 1000:
        raise SystemExit("the production USF protocol requires at least 1000 particles")
    if not 0.0 < args.volume_fraction < 0.1:
        raise SystemExit("volume fraction must lie in (0, 0.1)")
    replicates = args.replicates or (
        2 if args.mode in ("numerics", "response-validation") else 4)
    if not 1 <= replicates <= len(SEEDS):
        raise SystemExit(f"replicates must lie in [1,{len(SEEDS)}]")
    rows = campaign_rows(
        args.mode, Path(args.results), args.particles, args.volume_fraction,
        replicates, load_initialization_reference(Path(args.reference)))
    artifact = Path(args.artifact)
    if not artifact.is_file():
        raise SystemExit(f"artifact does not exist: {artifact}")
    with artifact.open("rb") as handle:
        artifact_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    for row in rows:
        row["protocol_version"] = "usf-crossflow-v1"
        row["artifact_sha256"] = artifact_hash
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} {args.mode} USF study tasks to {output}")


if __name__ == "__main__":
    main()
