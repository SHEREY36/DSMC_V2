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
    "seed", "particles", "initial_theta", "relaxation_horizon",
    "dissipation_horizon", "tau_end",
    "sample_start_tau", "sample_end_tau", "sample_delta_tau", "state_update_cpp", "dt",
    "orientation_integrator", "max_ntc_candidates_per_step", "output_prefix",
)
PROTOCOL_VERSION = "hcs-ng-v8"
ORIENTATION_INTEGRATOR = "symmetric_midpoint_v1"
# The v5 long-time sentinel resolved an A_cw discrepancy at alpha=0.5,
# AR=1.35 between 0.005 and 0.0025. Promote the converged arm and test it
# against another factor-of-two refinement before any full-domain allocation.
PRODUCTION_DT = 0.0025
CONVERGENCE_DT = 0.00125

# --- Protocol v8: the collision clock, measured -----------------------------
#
# The completed v7 148-task two-sided stability campaign measured, for the
# first time, how long the HCS attractor actually takes to form.  At all 37
# coordinates both initial branches (theta0 = 0.025/0.225 and 1.5) enter the
# late-time band within tau <= 130 collisions per particle; the median is 21
# and the 90th percentile 62, with the worst case at alpha=0.95, AR=1.1.
#
# Relaxation is therefore governed by the *collision* clock, not by
# accumulated cooling.  Expressed as chi = (1-alpha**2)*tau the same
# measurement spans only 6-41, so the v7 rule chi >= 600 over-ran the slowest
# coordinate by a factor of about 47 and cost tau_end = 6154 cpp at
# alpha=0.95 -- 20 h per task, past the hard-coded physical-time ceiling.
#
# v8 runs every long-time stage on the production window instead, which is
# also the window of Megias & Santos 2023 (s in [500, 1500], delta s = 5).
MEASURED_ATTRACTOR_RELAXATION_CPP = 130.0
ATTRACTOR_RELAXATION_MARGIN = 3.0
PRODUCTION_TAU_END = 1500.0
PRODUCTION_SAMPLE_START = 500.0
PRODUCTION_SAMPLE_DELTA = 5.0
# Production particle count.  At fixed box volume the number density scales
# with N, so the physical time needed to reach a given cpp scales as 1/N while
# the per-step cost scales as N: measured across five coordinates, N=10000
# costs only 1.1-1.5x the wall time of N=2000 for the same cpp while giving
# 5x the samples per snapshot.  The v7 stability screen at N=2000 carried
# per-snapshot noise (sd(a20) = 0.016-0.065) two to six times larger than the
# 0.01 drift it was asked to resolve, which is why it failed at random.
PRODUCTION_PARTICLES = 10000
assert PRODUCTION_SAMPLE_START >= (ATTRACTOR_RELAXATION_MARGIN
                                   * MEASURED_ATTRACTOR_RELAXATION_CPP)
SEEDS = (260916101, 260916211, 260916307, 260916419, 260916523,
         260916631, 260916733, 260916839, 260916947, 260917051,
         260917159, 260917267, 260917373, 260917481, 260917589,
         260917697)


# Measured on the v7 stability campaign: at alpha=0.50, AR=1.10 the HCS
# attractor sits at a20 ~ 0.140 and Trot/Ttr ~ 20, while the artifact's a2_tr
# support is [-0.086, +0.117].  Half of every evaluation-window closure query
# there falls back to the base law, so that coordinate measures the fallback,
# not the learned closure.  It is excluded from the production grid and
# reported as the model's validated-domain boundary.  AR=1.20 stays in: its
# mean a20 = 0.055 is well inside the hull and its 3-5 per cent v7 fallback
# was an N=2000 boundary excursion (SE(a20) = 0.028 at N=2000 against 0.013 at
# N=10000), which the v8 particle count is expected to remove.
ARTIFACT_DOMAIN_EXCLUSIONS = ((0.50, 1.10),)


def production_grid() -> tuple[tuple[float, float], ...]:
    """Paper-style cross design (cf. Megias & Santos 2023, Figs. 4-5 with AR
    in place of beta).  alpha sweeps at three AR, AR sweeps at the calibrated
    alpha nodes, with alpha=1 as the exact-equipartition (Gaussian) control."""
    alpha_sweep = tuple((a, ar) for ar in (1.5, 2.0, 3.0)
                        for a in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00))
    ar_sweep = tuple((a, ar) for a in (0.50, 0.80, 0.95, 1.00)
                     for ar in (1.10, 1.20, 1.35, 2.5))
    return tuple(case for case in alpha_sweep + ar_sweep
                 if case not in ARTIFACT_DOMAIN_EXCLUSIONS)


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
    if mode == "numerics-pilot":
        # Protocol v6 isolated its only numerical-control failure here.  Test
        # the repaired midpoint integrator at the exact production N/window
        # before spending another five-case engineering allocation.
        return ((0.50, 1.35),), SEEDS[:8], \
            ("scaled", "unscaled", "dt_half"), 10000, 60.0, 30.0, 2.0
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
        # Production science.  tau=500 is at least 3.8 measured relaxation
        # times past the slowest coordinate and 24 at the median, so the first
        # retained snapshot is already on the attractor.  v8 splits the ten
        # realizations across the two bracketing initial conditions instead of
        # starting them all from theta0=1: the two-sided attraction test then
        # rests on production statistics, and once the branches agree all ten
        # realizations pool for the cumulants at no statistical cost.
        return production_grid(), SEEDS[:5], ("scaled",), PRODUCTION_PARTICLES, \
            PRODUCTION_TAU_END, PRODUCTION_SAMPLE_START, PRODUCTION_SAMPLE_DELTA
    if mode in ("stability-sentinel", "stability"):
        # Every production coordinate must reach its HCS attractor from both
        # sides. v8 screens on the production window and particle count: the
        # v7 screen ran N=2000 on a chi=600 cooling horizon, which was both
        # 47x longer than the measured relaxation and too noisy to resolve the
        # 0.01 drift it tested. The sentinel keeps four seeds and the half-step
        # arm, because v5 showed two seeds cannot resolve the a02 time-step
        # control; the full-domain stage stays a two-seed screen.
        sentinels = ((0.50, 1.35), (0.50, 3.0), (0.80, 2.0),
                     (0.95, 2.0), (1.00, 3.0))
        if mode == "stability-sentinel":
            return sentinels, SEEDS[:4], ("scaled", "dt_half"), \
                PRODUCTION_PARTICLES, PRODUCTION_TAU_END, \
                PRODUCTION_SAMPLE_START, PRODUCTION_SAMPLE_DELTA
        return production_grid(), SEEDS[:2], ("scaled",), \
            PRODUCTION_PARTICLES, PRODUCTION_TAU_END, \
            PRODUCTION_SAMPLE_START, PRODUCTION_SAMPLE_DELTA
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
    parser.add_argument("--mode", choices=("numerics-pilot", "engineering",
                                            "domain-pilot", "sweep",
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
        low_theta = 0.025 if ar <= 1.35 else 0.225
        # Two-sided modes bracket every predicted production root (about 0.038
        # through one) from below and above. The low-theta extension exists
        # through AR=1.35; at larger AR the calibrated hull begins at theta=0.2.
        two_sided = args.mode in ("stability-sentinel", "stability", "sweep")
        initial_thetas = ((low_theta, 1.5) if two_sided else (1.0,))
        case_arms = arms
        for initial_theta in initial_thetas:
            for arm in case_arms:
                for replicate, seed in enumerate(seeds):
                    tau_value, start_value, delta_value = tau_end, start, delta
                    # v8 states the horizon on the collision clock that
                    # actually governs relaxation.  chi = (1-alpha**2)*tau is
                    # still reported, but it no longer sets the run length.
                    # Short numerical-control pilots make no attractor claim,
                    # so they declare no relaxation horizon.
                    horizon = (0.0 if args.mode in ("numerics-pilot",
                                                    "engineering")
                               else MEASURED_ATTRACTOR_RELAXATION_CPP
                               * ATTRACTOR_RELAXATION_MARGIN)
                    theta_tag = (f"_theta0_{initial_theta:.2f}"
                                 if two_sided else "")
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
                        "relaxation_horizon": horizon,
                        "dissipation_horizon": (
                            tau_value if alpha >= 1.0
                            else (1.0 - alpha**2) * tau_value),
                        "tau_end": tau_value,
                        "sample_start_tau": start_value, "sample_end_tau": tau_value,
                        "sample_delta_tau": delta_value,
                        "state_update_cpp": 0.05,
                        "orientation_integrator": ORIENTATION_INTEGRATOR,
                        # Protocol v2 rejected 0.01 and protocol v5 rejected
                        # 0.005 at the strongest-dissipation near-sphere
                        # sentinel. Production is now 0.0025, checked against
                        # 0.00125 at both short and long validation horizons.
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
