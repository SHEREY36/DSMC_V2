#!/usr/bin/env python3
"""Summarize paired USF trajectories, compare with DEM, and plot the gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("Pxx", "Pyy", "Pzz", "Pxy")


def relative_linear_drift(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    return abs(slope) * float(x[-1] - x[0]) / max(abs(float(np.mean(y))), 1e-12)


def load_references(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = (float(row["AR"]), float(row["alpha"]))
        result[key] = {name: float(row[name]) for name in (*COMPONENTS, "theta")}
    return result


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _s_from_orientation(data: np.ndarray) -> np.ndarray:
    q = np.zeros((len(data), 3, 3))
    q[:, 0, 0], q[:, 0, 1], q[:, 0, 2] = data[:, 2], data[:, 3], data[:, 4]
    q[:, 1, 0], q[:, 1, 1], q[:, 1, 2] = data[:, 3], data[:, 5], data[:, 6]
    q[:, 2, 0], q[:, 2, 1], q[:, 2, 2] = data[:, 4], data[:, 6], data[:, 7]
    return np.linalg.eigvalsh(q)[:, -1]


def analyse_run(row: dict[str, str]) -> tuple[dict, dict[str, np.ndarray]]:
    prefix = Path(row["output_prefix"])
    trajectory = np.atleast_2d(np.loadtxt(Path(str(prefix) + ".txt")))
    pressure = np.atleast_2d(np.loadtxt(Path(str(prefix) + "_pressure.txt")))
    orientation = np.atleast_2d(np.loadtxt(Path(str(prefix) + "_orientation.txt")))
    diagnostics = json.loads(Path(str(prefix) + ".json").read_text())
    if trajectory.shape[1] != 5 or pressure.shape[1] != 14 or orientation.shape[1] != 8:
        raise ValueError(f"unexpected USF output shape for {prefix}")

    tau = trajectory[:, 1]
    ttr, trot, total = trajectory[:, 2], trajectory[:, 3], trajectory[:, 4]
    theta = ttr / trot
    tail_tau = 0.70 * float(tau[-1])
    tail_t = tau >= tail_tau
    tail_p = pressure[:, 1] >= tail_tau
    tail_q = orientation[:, 1] >= tail_tau
    if min(np.count_nonzero(tail_t), np.count_nonzero(tail_p),
           np.count_nonzero(tail_q)) < 10:
        raise ValueError(f"insufficient late-time samples for {prefix}")

    ttr_at_pressure = np.interp(pressure[:, 1], tau, ttr)
    density = float(diagnostics["number_density"])
    kinetic = np.column_stack((pressure[:, 2], pressure[:, 5],
                               pressure[:, 7], pressure[:, 3]))
    collisional = np.column_stack((pressure[:, 8], pressure[:, 11],
                                   pressure[:, 13], pressure[:, 9]))
    reduced = (kinetic + collisional) / (density * ttr_at_pressure[:, None])
    reduced_kinetic = kinetic / (density * ttr_at_pressure[:, None])
    nematic = _s_from_orientation(orientation)

    means = {name: float(np.mean(reduced[tail_p, index]))
             for index, name in enumerate(COMPONENTS)}
    kinetic_means = {name: float(np.mean(reduced_kinetic[tail_p, index]))
                     for index, name in enumerate(COMPONENTS)}
    means.update({
        "theta": float(np.mean(theta[tail_t])),
        "total_temperature": float(np.mean(total[tail_t])),
        "nematic_order": float(np.mean(nematic[tail_q])),
    })
    record = {
        "task_id": int(row["task_id"]),
        "mode": row["mode"], "arm": row["arm"],
        "alpha": float(row["alpha"]),
        "aspect_ratio": float(row["aspect_ratio"]),
        "replicate": int(row["replicate"]),
        "seed": int(row["seed"]),
        "late_means": means,
        "late_kinetic_pressure_means": kinetic_means,
        "late_theta_relative_drift": relative_linear_drift(tau[tail_t], theta[tail_t]),
        "late_total_temperature_relative_drift": relative_linear_drift(
            tau[tail_t], total[tail_t]),
        "negative_energy_repairs": int(diagnostics["negative_energy_repairs"]),
        "energy_axis_clamps": int(diagnostics.get("energy_axis_clamps", 0)),
        "energy_monotonic_repairs": int(diagnostics.get("energy_monotonic_repairs", 0)),
        "out_of_domain_fraction": float(diagnostics["out_of_domain_fraction"]),
        "closure_overhead_fraction": float(diagnostics["closure_overhead_fraction"]),
        "runtime_seconds": float(diagnostics.get("runtime_seconds", 0.0)),
        "artifact": diagnostics.get("artifact"),
        "artifact_sha256": diagnostics.get("artifact_sha256"),
        "finite": bool(np.all(np.isfinite(trajectory))
                       and np.all(np.isfinite(pressure))
                       and np.all(np.isfinite(orientation))
                       and np.all(ttr > 0.0) and np.all(trot > 0.0)),
    }
    series = {"tau": tau, "theta": theta, "total": total,
              "pressure_tau": pressure[:, 1], "pressure": reduced,
              "orientation_tau": orientation[:, 1], "nematic": nematic}
    return record, series


def mean_curve_drift(items: list[tuple[dict, dict[str, np.ndarray]]],
                     quantity: str) -> float:
    length = min(len(series[quantity]) for _, series in items)
    tau = items[0][1]["tau"][:length]
    values = np.asarray([series[quantity][:length] for _, series in items])
    start = max(1, int(0.70 * length))
    return relative_linear_drift(tau[start:], np.mean(values, axis=0)[start:])


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference)
                 / max(np.linalg.norm(reference), 1e-12))


def summarize(rows: list[dict[str, str]], references: dict,
              loaded: list[tuple[dict, dict[str, np.ndarray]]]) -> dict:
    grouped: dict[tuple[str, float, float], list] = defaultdict(list)
    for item in loaded:
        record = item[0]
        grouped[(record["arm"], record["aspect_ratio"], record["alpha"])].append(item)

    cases = []
    for (arm, ar, alpha), items in sorted(grouped.items()):
        reference = references[(ar, alpha)]
        values = {name: np.array([item[0]["late_means"][name] for item in items])
                  for name in (*COMPONENTS, "theta", "nematic_order")}
        means = {name: float(np.mean(array)) for name, array in values.items()}
        stds = {name: float(np.std(array, ddof=1)) for name, array in values.items()}
        cvs = {name: stds[name] / max(abs(means[name]), 1e-12)
               for name in values}
        predicted_stress = np.array([means[name] for name in COMPONENTS])
        reference_stress = np.array([reference[name] for name in COMPONENTS])
        component_errors = {
            name: abs(means[name] - reference[name]) / max(abs(reference[name]), 1e-12)
            for name in COMPONENTS}
        stress_error = _relative_l2(predicted_stress, reference_stress)
        theta_error = abs(means["theta"] - reference["theta"]) / reference["theta"]
        theta_drift = mean_curve_drift(items, "theta")
        total_drift = mean_curve_drift(items, "total")
        physical_run_pass = all(
            item[0]["finite"]
            and item[0]["negative_energy_repairs"] == 0
            and item[0]["energy_axis_clamps"] == 0
            and item[0]["energy_monotonic_repairs"] == 0
            and item[0]["out_of_domain_fraction"] < 1e-3
            for item in items)
        stationarity_pass = theta_drift <= 0.10 and total_drift <= 0.10
        precision_pass = max(cvs[name] for name in (*COMPONENTS, "theta")) <= 0.10
        accuracy_pass = (stress_error <= 0.10 and theta_error <= 0.10
                         and max(component_errors.values()) <= 0.15)
        cases.append({
            "arm": arm, "aspect_ratio": ar, "alpha": alpha,
            "n_replicates": len(items), "means": means,
            "standard_deviations": stds,
            "coefficients_of_variation": cvs,
            "reference": reference,
            "component_relative_errors": component_errors,
            "stress_relative_l2_error": stress_error,
            "theta_relative_error": theta_error,
            "replicate_mean_theta_relative_drift": theta_drift,
            "replicate_mean_total_temperature_relative_drift": total_drift,
            "physical_run_pass": physical_run_pass,
            "stationarity_pass": stationarity_pass,
            "precision_pass": precision_pass,
            "accuracy_pass": accuracy_pass,
            "physics_pass": bool(physical_run_pass and stationarity_pass
                                 and precision_pass and accuracy_pass),
        })

    by_arm = {}
    for arm in sorted({case["arm"] for case in cases}):
        selected = [case for case in cases if case["arm"] == arm]
        prediction = np.array([[case["means"][name] for name in COMPONENTS]
                               for case in selected])
        reference = np.array([[case["reference"][name] for name in COMPONENTS]
                              for case in selected])
        theta_pred = np.array([case["means"]["theta"] for case in selected])
        theta_ref = np.array([case["reference"]["theta"] for case in selected])
        by_arm[arm] = {
            "n_cases": len(selected),
            "stress_relative_rmse": float(
                np.sqrt(np.mean((prediction - reference) ** 2))
                / np.sqrt(np.mean(reference**2))),
            "theta_relative_rmse": float(
                np.sqrt(np.mean((theta_pred - theta_ref) ** 2))
                / np.sqrt(np.mean(theta_ref**2))),
            "physics_pass": all(case["physics_pass"] for case in selected),
        }
    correction_improves_stress = (
        "uncorrected" not in by_arm
        or by_arm["corrected"]["stress_relative_rmse"]
        < by_arm["uncorrected"]["stress_relative_rmse"])
    corrected_pass = by_arm.get("corrected", {}).get("physics_pass", False)
    performance_pass = all(
        record["closure_overhead_fraction"] < 0.05 for record, _ in loaded
        if record["arm"] == "corrected")
    artifact_digests = {record["artifact_sha256"] for record, _ in loaded
                        if record.get("artifact_sha256")}
    artifact_consistent = bool(len(artifact_digests) == 1 and all(
        record.get("artifact_sha256") for record, _ in loaded))
    return {
        "schema_version": "usf_validation_v1",
        "criteria": {
            "tail_fraction": 0.30,
            "stress_relative_l2_error_max": 0.10,
            "individual_stress_relative_error_max": 0.15,
            "theta_relative_error_max": 0.10,
            "replicate_coefficient_of_variation_max": 0.10,
            "replicate_mean_relative_drift_max": 0.10,
            "out_of_domain_fraction_exclusive_maximum": 1e-3,
            "closure_overhead_fraction_exclusive_maximum": 0.05,
        },
        "n_expected": len(rows), "n_valid": len(loaded),
        "artifact_sha256": (next(iter(artifact_digests))
                            if artifact_consistent else None),
        "artifact_consistent": artifact_consistent,
        "runs": [record for record, _ in loaded],
        "cases": cases, "arms": by_arm,
        "correction_improves_stress": correction_improves_stress,
        "physics_gate_pass": bool(artifact_consistent and corrected_pass
                                  and correction_improves_stress),
        "production_gate_pass": bool(artifact_consistent and corrected_pass
                                     and correction_improves_stress and performance_pass),
        "full_validation_ready": bool(artifact_consistent and corrected_pass
                                      and correction_improves_stress),
        "deployment_ready": bool(
            rows and rows[0]["mode"] == "full" and artifact_consistent and corrected_pass
            and performance_pass),
    }


def plot_summary(summary: dict, loaded: list, figure: Path) -> None:
    colors = {"corrected": "#0072B2", "uncorrected": "#D55E00"}
    cases = summary["cases"]
    ars = sorted({case["aspect_ratio"] for case in cases})
    fig, axes = plt.subplots(len(ars), 3, figsize=(12, 3.4 * len(ars)),
                             squeeze=False)
    for row_index, ar in enumerate(ars):
        selected = [case for case in cases if case["aspect_ratio"] == ar]
        for arm in sorted({case["arm"] for case in selected}):
            family = sorted((case for case in selected if case["arm"] == arm),
                            key=lambda case: case["alpha"])
            alpha = [case["alpha"] for case in family]
            axes[row_index, 0].plot(
                alpha, [case["means"]["theta"] for case in family], "o-",
                color=colors[arm], label=arm)
            axes[row_index, 1].plot(
                alpha, [case["means"]["Pxy"] for case in family], "o-",
                color=colors[arm], label=arm)
            axes[row_index, 2].plot(
                alpha, [case["means"]["nematic_order"] for case in family], "o-",
                color=colors[arm], label=arm)
        reference_family = sorted(
            {case["alpha"]: case["reference"] for case in selected}.items())
        axes[row_index, 0].plot(
            [item[0] for item in reference_family],
            [item[1]["theta"] for item in reference_family], "ks--", label="DEM")
        axes[row_index, 1].plot(
            [item[0] for item in reference_family],
            [item[1]["Pxy"] for item in reference_family], "ks--", label="DEM")
        axes[row_index, 0].set_ylabel(fr"AR={ar:g}: $T_{{tr}}/T_{{rot}}$")
        axes[row_index, 1].set_ylabel(fr"AR={ar:g}: $P_{{xy}}^*$")
        axes[row_index, 2].set_ylabel(fr"AR={ar:g}: $S$")
        for ax in axes[row_index]:
            ax.set_xlabel(r"coefficient of restitution $\alpha$")
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle("Frozen-artifact uniform-shear validation", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", default=(
        "DSMC_0D_v2/reference/usf_dem_reference.csv"))
    parser.add_argument("--summary", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args()
    rows = load_manifest(Path(args.manifest))
    references = load_references(Path(args.reference))
    loaded, missing = [], []
    for row in rows:
        prefix = Path(row["output_prefix"])
        required = [Path(str(prefix) + suffix) for suffix in (
            ".txt", "_pressure.txt", "_orientation.txt", ".json")]
        if all(path.exists() for path in required):
            loaded.append(analyse_run(row))
        else:
            missing.append(int(row["task_id"]))
    if missing:
        raise SystemExit(f"missing USF outputs for tasks {missing}")
    summary = summarize(rows, references, loaded)
    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_summary(summary, loaded, Path(args.figure))
    print(json.dumps({key: summary[key] for key in (
        "n_expected", "n_valid", "arms", "correction_improves_stress",
        "physics_gate_pass", "production_gate_pass", "full_validation_ready",
        "deployment_ready")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
