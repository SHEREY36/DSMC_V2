#!/usr/bin/env python3
"""Generate the gated schema-2.2 CTC restart campaign manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SENTINEL_ALPHA = (0.5, 0.8, 0.95, 1.0)
SENTINEL_THETA = (0.2, 1.0, 2.0)
SENTINEL_AR = (1.1, 2.0, 3.0)
BASELINE_ALPHA = tuple(round(0.50 + 0.05 * i, 2) for i in range(10)) + (0.975, 0.99, 1.0)
BASELINE_THETA = tuple(round(0.1 * i, 1) for i in range(1, 13)) + (1.4, 1.6, 1.8, 2.0)
BASELINE_AR = (1.1, 1.25, 1.5, 2.0, 2.5, 3.0)
COARSE_ALPHA = (0.70, 0.85, 0.95)
COARSE_THETA = (0.2, 0.5, 1.0, 2.0)
COARSE_AR = (1.1, 2.0, 3.0)
FIELDS = ("task_id", "stage", "role", "alpha", "theta", "aspect_ratio",
          "ensemble_id", "ensemble_mode", "control", "seed", "nsamples",
          "shard", "output_directory")


def _ensembles() -> list[dict]:
    rows = [{"ensemble_id": 0, "ensemble_mode": "baseline", "control": "0"}]
    definitions = [
        ("a2_tr", ("s1.5", "s1.75", "s2.5", "s4.0")),
        ("a2_rot", ("s1.5", "s1.75", "s2.5", "s4.0")),
        ("a11", ("rho-0.6", "rho-0.3", "rho+0.3", "rho+0.6")),
        ("A_cu", ("k-2", "k-1", "k+1", "k+2")),
        ("PiPi", ("shear0.15", "shear0.30", "uni0.15", "uni0.30")),
        ("QQ", ("obl0.15", "obl0.30", "pro0.15", "pro0.30")),
        ("RtRt", ("x0.15", "x0.30", "z0.15", "z0.30")),
        ("PiQ", ("parallel0.15", "parallel0.30", "orthogonal0.15", "orthogonal0.30")),
        ("PiRt", ("parallel0.15", "parallel0.30", "orthogonal0.15", "orthogonal0.30")),
        ("QRt", ("parallel0.15", "parallel0.30", "orthogonal0.15", "orthogonal0.30")),
        ("heat_flux", ("tr", "rot", "parallel", "antiparallel")),
        ("W2", ("plus0.15", "plus0.30")),
    ]
    for mode, controls in definitions:
        for control in controls:
            rows.append({"ensemble_id": len(rows), "ensemble_mode": mode,
                         "control": control})
    if len(rows) != 47:
        raise AssertionError(f"excitation design must contain 47 ensembles, got {len(rows)}")
    return rows


ENSEMBLES = _ensembles()


def _seed(alpha_index: int, theta_index: int, ar_index: int,
          ensemble_id: int, shard: int) -> int:
    # Common random numbers are retained along alpha lines.
    del alpha_index
    return 260902 + 100000 * ar_index + 5000 * theta_index + 50 * ensemble_id + shard


def _row(task: int, stage: str, role: str, alpha: float, theta: float, ar: float,
         ensemble: dict, samples: int, shard: int, root: str,
         ai: int, ti: int, ri: int) -> dict:
    eid = ensemble["ensemble_id"]
    tag = (f"alpha_{alpha:.3f}_theta_{theta:.3f}_AR_{ar:.3f}_"
           f"ensemble_{eid:03d}_shard_{shard:02d}")
    return {"task_id": task, "stage": stage, "role": role, "alpha": alpha,
            "theta": theta, "aspect_ratio": ar, **ensemble,
            "seed": _seed(ai, ti, ri, eid, shard), "nsamples": samples,
            "shard": shard, "output_directory": f"{root}/{stage}/{tag}"}


def grid_rows(stage: str, samples: int, shard: int, root: str) -> list[dict]:
    rows = []
    if stage == "sentinel":
        axes, ensembles = (SENTINEL_ALPHA, SENTINEL_THETA, SENTINEL_AR), ENSEMBLES[:1]
        role = "baseline_sentinel"
    elif stage == "baseline":
        axes, ensembles = (BASELINE_ALPHA, BASELINE_THETA, BASELINE_AR), ENSEMBLES[:1]
        role = "baseline"
    elif stage == "excitation":
        axes, ensembles = (COARSE_ALPHA, COARSE_THETA, COARSE_AR), ENSEMBLES
        role = "direct_excitation"
    else:
        raise ValueError(stage)
    alphas, thetas, ars = axes
    for ri, ar in enumerate(ars):
        for ti, theta in enumerate(thetas):
            for ai, alpha in enumerate(alphas):
                for ensemble in ensembles:
                    rows.append(_row(len(rows), stage, role, alpha, theta, ar,
                                     ensemble, samples, shard, root, ai, ti, ri))
    if stage == "excitation":
        rows.append(_row(len(rows), stage, "elastic_sentinel", 1.0, 1.0, 2.0,
                         ENSEMBLES[0], samples, shard, root, 0, 0, 0))
    return rows


def adaptive_rows(report: str | Path, samples: int, shard: int, root: str) -> list[dict]:
    payload = json.loads(Path(report).read_text())
    rows = []
    for source in payload.get("continue", []):
        completed = int(source["n_outcomes"])
        if completed >= 200_000:
            continue
        ensemble = ENSEMBLES[int(source.get("ensemble_id", 0))]
        add = min(samples, 200_000 - completed)
        rows.append(_row(len(rows), "adaptive", "precision_continuation",
                         float(source["alpha"]), float(source["theta"]),
                         float(source["aspect_ratio"]), ensemble, add, shard, root,
                         0, 0, 0))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("sentinel", "baseline", "excitation", "adaptive"),
                        required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--results-root", default="results/ctc_closure")
    parser.add_argument("--precision-report")
    args = parser.parse_args()
    if not (1 <= args.samples <= 200_000):
        parser.error("--samples must lie in 1..200000")
    if args.stage == "adaptive":
        if not args.precision_report:
            parser.error("adaptive stage requires --precision-report")
        rows = adaptive_rows(args.precision_report, args.samples, args.shard, args.results_root)
    else:
        rows = grid_rows(args.stage, args.samples, args.shard, args.results_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} {args.stage} tasks to {output}")


if __name__ == "__main__":
    main()
