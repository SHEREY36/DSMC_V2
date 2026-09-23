#!/usr/bin/env python3
"""Create staged, one-realization-per-task non-Gaussian HCS designs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


FIELDS = (
    "task_id", "protocol_version", "mode", "arm", "model_variant",
    "invariant_corrections", "artifact_sha256", "alpha", "aspect_ratio", "replicate",
    "seed", "particles", "initial_theta", "dissipation_horizon", "tau_end",
    "sample_start_tau", "sample_end_tau", "sample_delta_tau", "state_update_cpp", "dt",
    "max_ntc_candidates_per_step", "output_prefix",
)
PROTOCOL_VERSION = "hcs-ng-v5"
PRODUCTION_DT = 0.005
CONVERGENCE_DT = 0.0025
LONG_TIME_DISSIPATION_HORIZON = 600.0
SEEDS = (260916101, 260916211, 260916307, 260916419, 260916523,
         260916631, 260916733, 260916839, 260916947, 260917051,
         260917159, 260917267, 260917373, 260917481, 260917589,
         260917697)


def surface_axes(artifact: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    with np.load(artifact, allow_pickle=False) as data:
        coordinates = np.asarray(data["surface_coordinates"], dtype=float)
    return (tuple(np.unique(coordinates[:, 0]).tolist()),
            tuple(np.unique(coordinates[:, 2]).tolist()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def design(mode: str, artifact: Path):
    if mode == "engineering":
        # Validate the similarity thermostat and dt before paying for the
        # sweep.  The two alpha=0.5 cases bracket shape at maximum cooling;
        # alpha=1 exercises the separate Hong-Morris elastic block.
        cases = ((0.50, 1.35), (0.50, 3.0), (0.80, 2.0),
                 (0.95, 2.0), (1.00, 3.0))
        # Use the production particle count: candidate concurrency and the
        # same-step particle-reuse rate depend on N.  Eight paired seeds make
        # the numerical-control checks sensitive at a declared resolution
        # without spending production-length trajectories on the pilot.
        return cases, SEEDS[:8], ("scaled", "unscaled", "dt_half"), \
            10000, 60.0, 30.0, 2.0
    if mode == "domain-pilot":
        alphas, ars = surface_axes(artifact)
        return tuple((a, ar) for a in alphas for ar in ars), SEEDS[:4], \
            ("scaled",), 4000, 200.0, 100.0, 5.0
    if mode == "sweep":
        # Paper-style cross design (cf. Megias & Santos 2023, Figs. 4-5 with
        # AR in place of beta).  alpha sweeps at three AR, AR sweeps at the
        # calibrated alpha nodes, with alpha=1 as the exact-equipartition
        # (Gaussian) control.  The long window includes AR=1.1 and 1.2: the
        # artifact's slowest predicted relaxation time is about 108 cpp, so
        # sampling from tau=500 gives more than four relaxation times before
        # the first retained snapshot and tau=1500 covers the near-sphere
        # part of the calibrated domain instead of silently deleting it.
        alpha_sweep = tuple((a, ar) for ar in (1.5, 2.0, 3.0)
                            for a in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00))
        ar_sweep = tuple((a, ar) for a in (0.50, 0.80, 0.95, 1.00)
                         for ar in (1.10, 1.20, 1.35, 2.5))
        return alpha_sweep + ar_sweep, SEEDS[:10], ("scaled",), 10000, \
            1500.0, 500.0, 5.0
    if mode in ("stability-sentinel", "stability"):
        # Every production coordinate must first survive a common amount of
        # accumulated cooling, chi=(1-alpha**2)*tau, from both sides of its HCS
        # attractor.  Two seeds resolve gross stochastic failures cheaply at
        # N=2000.  Run the five engineering sentinels at both dt values before
        # spending the full-domain allocation.
        sentinels = ((0.50, 1.35), (0.50, 3.0), (0.80, 2.0),
                     (0.95, 2.0), (1.00, 3.0))
        if mode == "stability-sentinel":
            return sentinels, SEEDS[:2], ("scaled", "dt_half"), 2000, \
                None, None, None
        alpha_sweep = tuple((a, ar) for ar in (1.5, 2.0, 3.0)
                            for a in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00))
        ar_sweep = tuple((a, ar) for a in (0.50, 0.80, 0.95, 1.00)
                         for ar in (1.10, 1.20, 1.35, 2.5))
        return alpha_sweep + ar_sweep, SEEDS[:2], ("scaled",), 2000, \
            None, None, None
    if mode == "map":
        _, ars = surface_axes(artifact)
        alphas = tuple(np.round(np.arange(0.50, 1.00, 0.05), 2))
        return tuple((a, ar) for a in alphas for ar in ars), SEEDS, \
            ("scaled",), 10000, 1500.0, 500.0, 5.0
    if mode == "tails":
        # Include an equal-statistics exact Gaussian negative control at every
        # shape; otherwise a generic straight-line tail fit can validate itself.
        cases = tuple((a, ar) for a in (0.50, 0.75, 0.95, 1.00)
                      for ar in (1.10, 2.00, 3.00))
        # One hundred independent realizations are generated deterministically
        # below; the first sixteen are shared with the map campaign.
        seeds = tuple(260916101 + 1009 * index for index in range(100))
        return cases, seeds, ("scaled",), 10000, 1500.0, 500.0, 5.0
    if mode == "sphere-controls":
        cases = tuple((a, 1.0) for a in (0.50, 0.75, 0.95, 1.00))
        return cases, SEEDS, ("sphere",), 10000, 1500.0, 500.0, 5.0
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("engineering", "domain-pilot", "sweep",
                                            "stability-sentinel", "stability", "map",
                                            "tails", "sphere-controls"),
                        default="engineering")
    parser.add_argument("--artifact", required=True)
    parser.add_argument(
        "--model-variant", choices=("angular_evidence", "baseline"),
        default="angular_evidence",
        help=("angular_evidence enables the validated angular-only response; "
              "baseline uses the same sampler with invariant response disabled"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    artifact = Path(args.artifact)
    if not artifact.is_file():
        raise SystemExit(f"missing artifact: {artifact}")
    artifact_hash = sha256(artifact)
    cases, seeds, arms, particles, tau_end, start, delta = design(args.mode, artifact)
    model_variant = ("sphere_exact" if args.mode == "sphere-controls"
                     else args.model_variant)
    corrections = model_variant == "angular_evidence"
    rows = []
    for alpha, ar in cases:
        # The low-theta extension exists through AR=1.35; at larger AR the
        # calibrated hull begins at theta=0.2. These supported starts bracket
        # every predicted production root (about 0.038 through one).
        low_theta = 0.025 if ar <= 1.35 else 0.225
        stability_mode = args.mode in ("stability-sentinel", "stability")
        initial_thetas = ((low_theta, 1.5) if stability_mode else (1.0,))
        case_arms = arms
        for initial_theta in initial_thetas:
            for arm in case_arms:
                for replicate, seed in enumerate(seeds):
                    if stability_mode:
                        if alpha < 1.0:
                            tau_value = LONG_TIME_DISSIPATION_HORIZON / (1.0 - alpha**2)
                            horizon = LONG_TIME_DISSIPATION_HORIZON
                        else:
                            tau_value = 1500.0
                            horizon = tau_value
                        start_value = 0.75 * tau_value
                        # Forty late-window snapshots are enough for a five-block
                        # change-point gate without creating production histograms.
                        delta_value = (tau_value - start_value) / 39.0
                    else:
                        tau_value, start_value, delta_value = tau_end, start, delta
                        horizon = (tau_value if alpha >= 1.0
                                   else (1.0 - alpha**2) * tau_value)
                    theta_tag = (f"_theta0_{initial_theta:.2f}"
                                 if stability_mode else "")
                    tag = (f"alpha_{alpha:.2f}_AR_{ar:.2f}{theta_tag}_{arm}_"
                           f"rep_{replicate:03d}")
                    rows.append({
                        "task_id": len(rows), "protocol_version": PROTOCOL_VERSION,
                        "mode": args.mode, "arm": arm,
                        "model_variant": model_variant,
                        "invariant_corrections": str(corrections).lower(),
                        "artifact_sha256": artifact_hash,
                        "alpha": f"{alpha:.2f}", "aspect_ratio": f"{ar:.2f}",
                        "replicate": replicate, "seed": seed,
                        "particles": particles, "initial_theta": initial_theta,
                        "dissipation_horizon": horizon, "tau_end": tau_value,
                        "sample_start_tau": start_value, "sample_end_tau": tau_value,
                        "sample_delta_tau": delta_value,
                        "state_update_cpp": 0.05,
                        # Protocol v2 found a statistically resolved a02 shift at
                        # AR=3 between dt=0.01 and 0.005. Production therefore
                        # uses 0.005, with 0.0025 controls at both the short and
                        # long validation horizons.
                        "dt": CONVERGENCE_DT if arm == "dt_half" else PRODUCTION_DT,
                        "max_ntc_candidates_per_step": max(100_000, 50 * particles),
                        "output_prefix": str(Path(args.results) / tag),
                    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} {args.mode} non-Gaussian HCS tasks to {output}")


if __name__ == "__main__":
    main()
