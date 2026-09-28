#!/usr/bin/env python3
"""Create the fixed, physics-first DSMC state-harvest design for USF response."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


# Every nonspherical artifact node is included. AR=1 is the separate exact
# sphere branch; the deliberately excluded latent interval is 1<AR<1.1.
ARS = (1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0)
ALPHAS = (0.50, 0.80, 0.95)
WINDOW_FRACTIONS = ((0.10, 0.20), (0.35, 0.45),
                    (0.60, 0.70), (0.85, 1.00))
BRANCHES = (("cold", 0.5, 0.75), ("hot", 2.0, 1.25))
SHEAR_RATES = {
    0.50: 0.0800, 0.55: 0.0771, 0.60: 0.0739, 0.65: 0.0702,
    0.70: 0.0660, 0.75: 0.0611, 0.80: 0.0554, 0.85: 0.0487,
    0.90: 0.0403, 0.95: 0.0288,
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_initialization_reference(path: Path) -> dict:
    result: dict[float, dict[float, tuple[float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(float(row["alpha"]), {})[float(row["AR"])] = (
                float(row["Tstar"]), float(row["theta"]))
    return result


def initialization_guess(reference: dict, ar: float,
                         alpha: float) -> tuple[float, float]:
    alphas = np.asarray(sorted(reference), dtype=float)
    tstar, theta = [], []
    for node_alpha in alphas:
        family = reference[float(node_alpha)]
        ars = np.asarray(sorted(family), dtype=float)
        tstar.append(np.interp(ar, ars, [family[x][0] for x in ars]))
        theta.append(np.interp(ar, ars, [family[x][1] for x in ars]))
    return (float(np.interp(alpha, alphas, tstar)),
            float(np.interp(alpha, alphas, theta)))


def shear_rate(alpha: float) -> float:
    nodes = np.asarray(sorted(SHEAR_RATES), dtype=float)
    return float(np.interp(alpha, nodes, [SHEAR_RATES[x] for x in nodes]))


def tau_end(alpha: float) -> float:
    return 160.0 if alpha <= 0.80 else (240.0 if alpha <= 0.90 else 360.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--artifact", default=(
        "models/microscopic_closure_v2_angular_evidence/closure_v2.npz"))
    parser.add_argument("--reference", default=(
        "DSMC_0D_v2/reference/usf_dem_fresh_v1.csv"))
    parser.add_argument("--particles", type=int, default=50_000)
    parser.add_argument("--reservoir-capacity", type=int, default=200_000)
    parser.add_argument("--volume-fraction", type=float, default=0.01)
    args = parser.parse_args()
    if args.particles < 20_000:
        raise SystemExit("response harvest requires at least 20,000 particles")
    if args.reservoir_capacity < 100_000:
        raise SystemExit("response harvest requires at least 100,000 pairs per window")

    artifact, reference = Path(args.artifact), Path(args.reference)
    if not artifact.is_file() or not reference.is_file():
        raise SystemExit("frozen artifact and initialization reference are required")
    artifact_sha256 = digest(artifact)
    reference_sha256 = digest(reference)
    reference_values = load_initialization_reference(reference)
    rows: list[dict[str, object]] = []
    results = Path(args.results)
    for ar in ARS:
        for alpha in ALPHAS:
            end = tau_end(alpha)
            base_tstar, base_theta = initialization_guess(
                reference_values, ar, alpha)
            rate = shear_rate(alpha)
            for branch, temperature_scale, theta_scale in BRANCHES:
                ttr = base_tstar * rate * rate * temperature_scale
                theta = base_theta * theta_scale
                prefix = results / (
                    f"AR_{ar:.3f}_alpha_{alpha:.3f}_{branch}")
                row = {
                    "task_id": len(rows),
                    "protocol_version": "usf-direct-response-v1",
                    "tag": args.tag,
                    "alpha": f"{alpha:.3f}",
                    "aspect_ratio": f"{ar:.3f}",
                    "initial_branch": branch,
                    "initial_ttr": f"{ttr:.12g}",
                    "initial_trot": f"{ttr / theta:.12g}",
                    "shear_rate": f"{rate:.12g}",
                    "dt": "0.005",
                    "tau_end": f"{end:.1f}",
                    "particles": args.particles,
                    "volume_fraction": f"{args.volume_fraction:.12g}",
                    "reservoir_capacity": args.reservoir_capacity,
                    "windows": ";".join(
                        f"{lo * end:.8g}:{hi * end:.8g}"
                        for lo, hi in WINDOW_FRACTIONS),
                    "seed": 280901001 + 1000 * len(rows),
                    "artifact": str(artifact),
                    "artifact_sha256": artifact_sha256,
                    # DEM is never a label.  Its steady values only reduce
                    # transient walltime; cold/hot convergence remains a gate.
                    "initialization_reference": str(reference),
                    "initialization_reference_sha256": reference_sha256,
                    "output_prefix": str(prefix),
                }
                rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} USF response-harvest tasks to {output}")


if __name__ == "__main__":
    main()
