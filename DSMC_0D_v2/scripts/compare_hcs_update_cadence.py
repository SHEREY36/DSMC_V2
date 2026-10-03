#!/usr/bin/env python3
"""Validate a collision-based closure refresh against every-step reference runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


IDENTITY = ("alpha", "aspect_ratio", "theta0", "replicate", "seed")


def rows(path: str) -> dict[tuple[str, ...], dict[str, str]]:
    with Path(path).open(newline="") as handle:
        values = list(csv.DictReader(handle))
    return {tuple(row[name] for name in IDENTITY): row for row in values}


def load(row: dict[str, str]) -> tuple[np.ndarray, dict]:
    prefix = Path(row["output_prefix"])
    return (np.atleast_2d(np.loadtxt(str(prefix) + ".txt")),
            json.loads(Path(str(prefix) + ".json").read_text()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--optimized-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference, optimized = rows(args.reference_manifest), rows(args.optimized_manifest)
    if set(reference) != set(optimized):
        raise SystemExit("reference and optimized HCS designs do not match")

    paired = defaultdict(list)
    for identity in sorted(reference):
        exact, exact_diag = load(reference[identity])
        fast, fast_diag = load(optimized[identity])
        stop = min(float(exact[-1, 1]), float(fast[-1, 1]))
        tau = fast[fast[:, 1] <= stop, 1]
        if len(tau) < 10:
            raise SystemExit(f"insufficient overlapping trajectory for {identity}")
        exact_ttr = np.interp(tau, exact[:, 1], exact[:, 2])
        exact_trot = np.interp(tau, exact[:, 1], exact[:, 3])
        exact_theta = exact_ttr / exact_trot
        fast_rows = fast[:len(tau)]
        fast_theta = fast_rows[:, 2] / fast_rows[:, 3]
        exact_total = np.interp(tau, exact[:, 1], exact[:, 4]) / exact[0, 4]
        fast_total = fast_rows[:, 4] / fast[0, 4]
        artifact_match = (exact_diag.get("artifact_sha256")
                          == fast_diag.get("artifact_sha256")
                          and bool(exact_diag.get("artifact_sha256")))
        paired[identity[:3]].append({
            "tau": tau, "exact_theta": exact_theta, "fast_theta": fast_theta,
            "exact_total": exact_total, "fast_total": fast_total,
            "reference": exact_diag, "optimized": fast_diag,
            "artifact_match": artifact_match})
    cases = []
    for identity, replicates in sorted(paired.items()):
        length = min(len(item["tau"]) for item in replicates)
        tau = replicates[0]["tau"][:length]
        exact_theta = np.mean([np.interp(
            tau, item["tau"], item["exact_theta"]) for item in replicates], axis=0)
        fast_theta = np.mean([np.interp(
            tau, item["tau"], item["fast_theta"]) for item in replicates], axis=0)
        exact_total = np.mean([np.interp(
            tau, item["tau"], item["exact_total"]) for item in replicates], axis=0)
        fast_total = np.mean([np.interp(
            tau, item["tau"], item["fast_total"]) for item in replicates], axis=0)
        late = max(1, int(0.7 * length))
        theta_rmse = float(np.sqrt(np.mean((fast_theta - exact_theta) ** 2))
                           / max(abs(float(np.mean(exact_theta))), 1.0e-12))
        total_rmse = float(np.sqrt(np.mean((fast_total - exact_total) ** 2)))
        late_difference = float(abs(np.mean(fast_theta[late:])
                                    - np.mean(exact_theta[late:]))
                                / max(abs(float(np.mean(exact_theta[late:]))), 1.0e-12))
        exact_runtime = sum(item["reference"]["runtime_seconds"] for item in replicates)
        fast_runtime = sum(item["optimized"]["runtime_seconds"] for item in replicates)
        exact_updates = sum(item["reference"]["closure_state_updates"] for item in replicates)
        fast_updates = sum(item["optimized"]["closure_state_updates"] for item in replicates)
        speedup = float(exact_runtime / max(fast_runtime, 1.0e-30))
        update_reduction = float(exact_updates / max(fast_updates, 1))
        overhead = max(item["optimized"]["closure_overhead_fraction"]
                       for item in replicates)
        artifact_match = all(item["artifact_match"] for item in replicates)
        runtime_gates_pass = all(
            item["optimized"].get("runtime_gate", {}).get("pass", False)
            and item["reference"].get("negative_energy_repairs", 0) == 0
            and item["optimized"].get("negative_energy_repairs", 0) == 0
            and item["reference"].get("energy_axis_clamps", 0) == 0
            and item["optimized"].get("energy_axis_clamps", 0) == 0
            for item in replicates)
        passed = bool(artifact_match and theta_rmse <= 0.05
                      and total_rmse <= 0.02 and late_difference <= 0.03
                      and overhead < 0.15 and speedup >= 2.0
                      and update_reduction >= 10.0 and runtime_gates_pass)
        cases.append({
            "identity": dict(zip(IDENTITY[:3], identity)),
            "n_replicates": len(replicates),
            "relative_theta_rmse": theta_rmse,
            "normalized_total_energy_rmse": total_rmse,
            "late_theta_relative_difference": late_difference,
            "runtime_speedup": speedup,
            "state_update_reduction": update_reduction,
            "optimized_overhead_fraction_maximum": overhead,
            "optimized_runtime_gates_pass": runtime_gates_pass,
            "artifact_match": artifact_match, "pass": passed})
    payload = {
        "validation_contract": "paired-hcs-state-update-cadence-v1",
        "criteria": {"replicate_mean_relative_theta_rmse_max": 0.05,
                     "normalized_total_energy_rmse_max": 0.02,
                     "late_theta_relative_difference_max": 0.03,
                     "optimized_closure_overhead_fraction_max": 0.15,
                     "runtime_speedup_min": 2.0,
                     "state_update_reduction_min": 10.0},
        "n_cases": len(cases),
        "maximum_relative_theta_rmse": max(x["relative_theta_rmse"] for x in cases),
        "maximum_normalized_total_energy_rmse": max(
            x["normalized_total_energy_rmse"] for x in cases),
        "maximum_late_theta_relative_difference": max(
            x["late_theta_relative_difference"] for x in cases),
        "maximum_optimized_overhead_fraction": max(
            x["optimized_overhead_fraction_maximum"] for x in cases),
        "minimum_runtime_speedup": min(x["runtime_speedup"] for x in cases),
        "minimum_state_update_reduction": min(
            x["state_update_reduction"] for x in cases),
        "cadence_validation_pass": all(x["pass"] for x in cases),
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"},
                     indent=2, sort_keys=True))
    if not payload["cadence_validation_pass"]:
        raise SystemExit("optimized closure update cadence failed paired HCS validation")


if __name__ == "__main__":
    main()
