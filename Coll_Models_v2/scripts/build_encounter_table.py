#!/usr/bin/env python3
"""Measure the collision (encounter) cross-section sigma(theta, AR) from CTC shards.

For every shard the generator stages a pair beyond reach, draws an impact point
uniformly on a square of side 2 b_max (b_max = 1.01 (L + D)), and integrates the
force-free approach; any contact is a hit.  The encounter cross-section is

    sigma_enc = A_prop * N_hit / N_att,      A_prop = (2 b_max)^2,

the same estimator the v1 generator wrote to csx.txt.  No force acts before the
first contact, so sigma_enc cannot depend on alpha; shards are pooled over alpha
at each (theta, AR).  The table stores sigma_enc / (pi d^2) and the dynamic
factor D = sigma_enc / A_bar, where A_bar = pi (d^2 + d L + L^2/8) is the exact
isotropic mean projected excluded area.  D (1.00 to 1.35) carries the rotation
of the rods during the approach and is what the runtime interpolates in
(log theta, AR); the geometry is multiplied back analytically.

Mean contacts per encounter <k>(alpha, theta, AR) are stored per node for
converting DSMC encounter counts to DEM contact counts.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from dsmc_v2_contracts.io import load_run


def mean_projected_area(aspect_ratio: float, diameter: float = 1.0) -> float:
    length = (aspect_ratio - 1.0) * diameter
    return math.pi * (diameter**2 + diameter * length + length**2 / 8.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("shard_globs", nargs="+")
    args = parser.parse_args()
    area = defaultdict(lambda: [0.0, 0, 0, set()])   # (AR, theta) -> [A_prop, hits, attempts, sources]
    contacts = defaultdict(lambda: [0, 0])            # (alpha, theta, AR) -> [sum k, encounters]
    seen_streams, seen_outcomes = set(), set()
    for pattern in args.shard_globs:
        for directory in sorted(glob.glob(pattern)):
            if not (os.path.isfile(os.path.join(directory, "metadata_v2.json"))
                    and os.path.isfile(os.path.join(directory, "outcomes_v2.bin"))):
                continue
            run = load_run(directory)
            md = run.metadata
            if str(md.get("sampling_mode", "")).startswith("dsmc_post_ntc"):
                continue                       # USF replays are not Maxwellian nodes
            if float(md.get("diameter", 1.0)) != 1.0:
                raise SystemExit(f"{directory}: expected d = 1 shards")
            ar, theta, alpha = (round(float(md["aspect_ratio"]), 4),
                                round(float(md["theta"]), 5), round(float(md["alpha"]), 4))
            hits, attempts = len(run.outcomes), len(run.attempts)
            proposal = float(md["proposal_area"])
            expected = (2.0 * 1.01 * ar) ** 2    # L + D = AR d with d = 1
            if abs(proposal / expected - 1.0) > 1.0e-6:
                raise SystemExit(f"{directory}: proposal area {proposal} != (2 b_max)^2")
            # Shards at different alpha share one attempt stream (common random
            # numbers), and some directories are copies of others.  Count each
            # distinct stream once so the pooled estimate and its error are honest.
            stream = (int(md.get("seed", -1)), ar, theta, attempts, hits)
            if stream not in seen_streams:
                seen_streams.add(stream)
                cell = area[(ar, theta)]
                cell[0] = proposal
                cell[1] += hits
                cell[2] += attempts
            cell[3].add(directory.split(os.sep)[1] if os.sep in directory else directory)
            k = np.asarray(run.outcomes["n_contact"])
            outcome_key = (int(md.get("seed", -1)), alpha, ar, theta, len(k), int(k.sum()))
            if outcome_key not in seen_outcomes:      # skip copied directories
                seen_outcomes.add(outcome_key)
                contacts[(alpha, theta, ar)][0] += int(k.sum())
                contacts[(alpha, theta, ar)][1] += len(k)
    table = {}
    for ar in sorted({key[0] for key in area}):
        thetas = sorted(theta for (a, theta) in area if a == ar)
        sigma = [area[(ar, t)][0] * area[(ar, t)][1] / area[(ar, t)][2] for t in thetas]
        hits = [area[(ar, t)][1] for t in thetas]
        # binomial standard error of the hit fraction
        se = [area[(ar, t)][0] * math.sqrt(p * (1 - p) / area[(ar, t)][2])
              for t, p in zip(thetas, (area[(ar, t)][1] / area[(ar, t)][2] for t in thetas))]
        table[f"{ar:.4f}"] = {
            "theta": thetas,
            "sigma_enc_over_pi_d2": [round(s / math.pi, 6) for s in sigma],
            "sigma_enc_standard_error_over_pi_d2": [round(e / math.pi, 6) for e in se],
            "dynamic_factor": [round(s / mean_projected_area(ar), 6) for s in sigma],
            "encounters": hits,
        }
        print(f"AR {ar}: " + ", ".join(f"theta={t:g}: {s / math.pi:.4f} pi d^2"
                                          for t, s in zip(thetas, sigma)))
    payload = {
        "schema": "encounter-cross-section-v2",
        "definition": ("sigma_enc = (2 b_max)^2 N_hit / N_att from HS_CTC_v2 shards (d = 1), "
                       "pooled over alpha; dynamic_factor = sigma_enc / (pi (d^2 + d L + L^2/8)). "
                       "Measured from this project's CTC data; replaces the v1 AR polynomial "
                       "for the encounter event unit."),
        "interpolation": "dynamic_factor linear in log(theta) (flat outside), then linear in AR",
        "sources": sorted({s for cell in area.values() for s in cell[3]}),
        "cross_section": table,
        "contacts_per_encounter": [
            {"alpha": a, "theta": t, "AR": r, "k_mean": round(v[0] / v[1], 6), "encounters": v[1]}
            for (a, t, r), v in sorted(contacts.items())],
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=1)


if __name__ == "__main__":
    main()
