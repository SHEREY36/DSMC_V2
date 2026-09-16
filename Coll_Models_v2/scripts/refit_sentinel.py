#!/usr/bin/env python3
"""Re-fit the closure sentinel with the conditional I-projection energy kernel.

Writes node estimates to a fresh directory and prints a side-by-side comparison
with the previous gated-Beta estimates, so nothing existing is overwritten.

    hpc/python.sh Coll_Models_v2/scripts/refit_sentinel.py \
        --runs results/ctc_closure/sentinel \
        --output results/closure_estimates/sentinel_v2 \
        --previous results/closure_estimates/sentinel
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from coll_models_v2.pipeline import estimate_grid


def _key(row: dict) -> tuple[float, float, float, int]:
    return (row["alpha"], row["theta"], row["aspect_ratio"], row["ensemble_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/ctc_closure/sentinel")
    parser.add_argument("--output", default="results/closure_estimates/sentinel_v2")
    parser.add_argument("--previous", default="results/closure_estimates/sentinel")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--propensity-offsets", type=int, default=128,
                        help="0 keeps the static projected-area weight")
    arguments = parser.parse_args()

    started = time.time()
    results = estimate_grid(arguments.runs, arguments.output,
                            n_bootstrap=arguments.bootstrap,
                            propensity_offsets=arguments.propensity_offsets or None)
    elapsed = time.time() - started

    previous: dict[tuple, dict] = {}
    for path in sorted(Path(arguments.previous).glob("*.json")):
        if path.name.endswith("_report.json"):
            continue
        row = json.loads(path.read_text())
        previous[_key(row)] = row

    header = (f"{'AR':>5}{'theta':>7}{'alpha':>7} | {'p_exch':>8}{'lam1':>9}{'lam2':>9}"
              f"{'lam3':>9} | {'stat mu':>9} | {'bias%':>7}{'bal|z|':>8}{'ESS/N':>7} | "
              f"{'old fit':>10}{'new gates':>26}")
    print(header)
    print("-" * len(header))
    failures = []
    for row in sorted(results, key=lambda r: (r["aspect_ratio"], r["theta"], r["alpha"])):
        energy, qa = row["energy"], row["qa"]
        old = previous.get(_key(row), {})
        old_state = "infeasible" if "energy" not in old else "feasible"
        reasons = [name.removesuffix("_pass") for name in
                   ("energy_projection_pass", "model_form_pass", "elastic_pass",
                    "propensity_pass", "proposal_balance_pass", "ess_pass")
                   if not qa.get(name, False)]
        measure = row["measure"]
        bias = 100.0 * max(abs(p["relative_bias"]) for p in measure["propensity"])
        balance = max(b["maximum_absolute_z_score"] for b in measure["proposal_balance"])
        print(f"{row['aspect_ratio']:>5}{row['theta']:>7}{row['alpha']:>7} | "
              f"{energy['p_exch']:8.4f}{energy['lambda1']:9.3f}{energy['lambda2']:9.3f}"
              f"{energy['lambda3']:9.3f} | {energy['stationary_mean']:9.4f} | "
              f"{bias:7.2f}{balance:8.2f}{measure['ess_fraction']:7.3f} | {old_state:>10}"
              f"{('PASS' if not reasons else ';'.join(reasons)):>26}")
        if reasons:
            failures.append((_key(row), reasons))

    print()
    print(f"{len(results) - len(failures)}/{len(results)} nodes pass every sentinel gate")
    print(f"energy projection converged at "
          f"{sum(r['energy']['projection_converged'] for r in results)}/{len(results)} nodes")
    print(f"elapsed {elapsed / 60.0:.1f} min")
    for key, reasons in failures:
        print(f"   fail {key}: {', '.join(reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
