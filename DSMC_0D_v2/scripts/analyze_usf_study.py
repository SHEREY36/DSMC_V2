#!/usr/bin/env python3
"""Analyze the staged USF cross-flow study without calibrating to DEM."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PK_COMPONENTS = ("Pk_xx", "Pk_yy", "Pk_zz", "Pk_xy")
PROTOCOL = "usf-crossflow-v1"
ANALYSIS_REVISION = "usf-crossflow-analysis-v2"
SUPPORT_POLICY = "adaptive_base_law_v1"
FAIL_METRIC = 1.0e300
FALLBACK_REASON = (
    "correction_fallback_fraction_in_evaluation_window_not_below_0.01")
THRESHOLDS = {
    "correction_support_fraction_exclusive_maximum": 0.01,
    "closure_overhead_fraction_exclusive_maximum": 0.15,
    "bulk_to_thermal_temperature_ratio_exclusive_maximum": 1.0e-12,
    "energy_ledger_relative_residual_maximum": 5.0e-8,
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_relative_drift(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or not np.all(np.isfinite(y)):
        return FAIL_METRIC
    slope = float(np.polyfit(x, y, 1)[0])
    return abs(slope) * float(x[-1] - x[0]) / max(abs(float(np.mean(y))), 1e-12)


def relative_gap(a: float, b: float) -> float:
    return abs(a - b) / max(0.5 * (abs(a) + abs(b)), 1e-12)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def load_reference(path: Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    result = {}
    for row in rows(path):
        key = tuple(float(row[name]) for name in key_fields)
        result[key] = {name: float(value) for name, value in row.items()
                       if name not in key_fields and value not in (None, "")
                       and name not in ("flags",)}
    return result


def interpolate_dem(reference: dict, ar: float, alpha: float) -> dict[str, float]:
    fields = ("Tstar", "theta", *PK_COMPONENTS)
    alphas = sorted({key[1] for key in reference})
    result = {}
    for field in fields:
        at_alpha = []
        for node_alpha in alphas:
            family = sorted((node_ar, values[field])
                            for (node_ar, a), values in reference.items()
                            if a == node_alpha)
            at_alpha.append(float(np.interp(
                ar, [item[0] for item in family], [item[1] for item in family])))
        result[field] = float(np.interp(alpha, alphas, at_alpha))
    return result


def nematic_order(orientation: np.ndarray) -> np.ndarray:
    q = np.zeros((len(orientation), 3, 3))
    q[:, 0, 0], q[:, 0, 1], q[:, 0, 2] = (
        orientation[:, 2], orientation[:, 3], orientation[:, 4])
    q[:, 1, 0], q[:, 1, 1], q[:, 1, 2] = (
        orientation[:, 3], orientation[:, 5], orientation[:, 6])
    q[:, 2, 0], q[:, 2, 1], q[:, 2, 2] = (
        orientation[:, 4], orientation[:, 6], orientation[:, 7])
    return np.linalg.eigvalsh(q)[:, -1]


def analyze_run(row: dict[str, str]) -> dict:
    prefix = Path(row["output_prefix"])
    paths = {
        "trajectory": Path(str(prefix) + ".txt"),
        "pressure": Path(str(prefix) + "_pressure.txt"),
        "orientation": Path(str(prefix) + "_orientation.txt"),
        "energy": Path(str(prefix) + "_energy.txt"),
        "diagnostics": Path(str(prefix) + ".json"),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    trajectory = np.atleast_2d(np.loadtxt(paths["trajectory"]))
    pressure = np.atleast_2d(np.loadtxt(paths["pressure"]))
    orientation = np.atleast_2d(np.loadtxt(paths["orientation"]))
    energy = np.atleast_2d(np.loadtxt(paths["energy"]))
    diagnostics = json.loads(paths["diagnostics"].read_text())
    if (trajectory.shape[1] != 5 or pressure.shape[1] != 14
            or orientation.shape[1] != 8 or energy.shape[1] != 6):
        raise ValueError(f"unexpected output schema for {prefix}")

    tau = trajectory[:, 1]
    ttr, trot = trajectory[:, 2], trajectory[:, 3]
    start = float(row["evaluation_start_tau"])
    masks = {
        "trajectory": tau >= start,
        "pressure": pressure[:, 1] >= start,
        "orientation": orientation[:, 1] >= start,
        "energy": energy[:, 1] >= start,
    }
    if min(int(np.count_nonzero(mask)) for mask in masks.values()) < 20:
        raise ValueError(f"insufficient evaluation-window samples for {prefix}")
    sphere = row["sphere"].lower() == "true"
    shear_rate = float(row["shear_rate"])
    validation_case = diagnostics.get("validation_case") or {}
    mass = float(validation_case.get("particle_mass", 1.0))
    diameter = float(validation_case.get("particle_diameter", 1.0))
    tstar = ttr / (mass * (shear_rate * diameter) ** 2)
    theta = np.full_like(ttr, np.nan) if sphere else ttr / trot

    ttr_at_p = np.interp(pressure[:, 1], tau, ttr)
    density = float(diagnostics["number_density"])
    kinetic = np.column_stack((pressure[:, 2], pressure[:, 5],
                               pressure[:, 7], pressure[:, 3]))
    collisional = np.column_stack((pressure[:, 8], pressure[:, 11],
                                   pressure[:, 13], pressure[:, 9]))
    pk = kinetic / (density * ttr_at_p[:, None])
    # The rod branch vector is not available in the scalar-v1 collision API;
    # therefore this approximate total is never used as a release gate.
    p_approx = (kinetic + collisional) / (density * ttr_at_p[:, None])
    tail = masks["trajectory"]
    tail_p = masks["pressure"]
    tail_q = masks["orientation"]
    tail_e = masks["energy"]
    means = {
        "Tstar": float(np.mean(tstar[tail])),
        "nematic_order": None if sphere else float(np.mean(nematic_order(
            orientation)[tail_q])),
    }
    if not sphere:
        means["theta"] = float(np.mean(theta[tail]))
    for index, name in enumerate(PK_COMPONENTS):
        means[name] = float(np.mean(pk[tail_p, index]))
        means[name.replace("Pk_", "Papprox_")] = float(
            np.mean(p_approx[tail_p, index]))

    e = energy[tail_e]
    delta_energy = float(e[-1, 2] - e[0, 2])
    delta_shear = float(e[-1, 3] - e[0, 3])
    delta_collision = float(e[-1, 4] - e[0, 4])
    power_scale = max(abs(delta_shear), abs(delta_collision), 1e-12)
    ledger = diagnostics.get("usf_energy_ledger") or {}
    correction_fallback = float(diagnostics.get(
        "correction_fallback_fraction_in_evaluation_window", 0.0))
    arm = row["arm"]
    correction_support = correction_fallback < THRESHOLDS[
        "correction_support_fraction_exclusive_maximum"]
    fallback_policy = str(diagnostics.get("correction_fallback_policy", ""))
    runtime = diagnostics.get("runtime_gate") or {}
    runtime_reasons = set(runtime.get("reasons") or [])
    non_support_runtime_reasons = runtime_reasons - {FALLBACK_REASON}
    adaptive_fallback = bool(
        correction_support or arm != "corrected"
        or fallback_policy == "base_law")
    closure_overhead = float(diagnostics.get("closure_overhead_fraction", 0.0))
    bulk_fraction = float(diagnostics.get(
        "maximum_bulk_to_thermal_temperature_ratio", 0.0))
    repeated = float((diagnostics.get("ntc") or {}).get(
        "repeated_particle_pair_fraction", 0.0))
    finite = bool(
        np.all(np.isfinite(trajectory)) and np.all(np.isfinite(pressure))
        and np.all(np.isfinite(orientation)) and np.all(np.isfinite(energy))
        and np.all(ttr > 0.0) and (sphere or np.all(trot > 0.0)))
    physical = bool(
        finite and diagnostics.get("termination_reason") == "collision_target"
        and int(diagnostics.get("particles", -1)) == int(row["particles"])
        and int(diagnostics.get("negative_energy_repairs", -1)) == 0
        and int(diagnostics.get("energy_axis_clamps", -1)) == 0
        and int(diagnostics.get("energy_monotonic_repairs", -1)) == 0
        and not non_support_runtime_reasons
        and adaptive_fallback
        and closure_overhead
        < THRESHOLDS["closure_overhead_fraction_exclusive_maximum"]
        and bulk_fraction
        < THRESHOLDS["bulk_to_thermal_temperature_ratio_exclusive_maximum"]
        and float(ledger.get("relative_residual", FAIL_METRIC))
        <= THRESHOLDS["energy_ledger_relative_residual_maximum"])
    return {
        "task_id": int(row["task_id"]), "mode": row["mode"],
        "coordinate_role": row["coordinate_role"], "arm": row["arm"],
        "aspect_ratio": float(row["aspect_ratio"]),
        "alpha": float(row["alpha"]), "dt": float(row["dt"]),
        "rate_scale": float(row["rate_scale"]),
        "initial_branch": row["initial_branch"],
        "replicate": int(row["replicate"]), "means": means,
        "Tstar_relative_drift": finite_relative_drift(tau[tail], tstar[tail]),
        "theta_relative_drift": (None if sphere else
                                  finite_relative_drift(tau[tail], theta[tail])),
        "steady_energy_power_imbalance": abs(delta_shear + delta_collision)
        / power_scale,
        "energy_ledger_increment_residual": abs(
            delta_energy - delta_shear - delta_collision) / power_scale,
        "energy_ledger_relative_residual": float(ledger.get(
            "relative_residual", FAIL_METRIC)),
        "runtime_gate_reasons": sorted(runtime_reasons),
        "non_support_runtime_gate_reasons": sorted(
            non_support_runtime_reasons),
        "correction_support_pass": correction_support,
        "adaptive_base_law_pass": adaptive_fallback,
        "correction_fallback_policy": fallback_policy,
        "correction_fallback_fraction_in_evaluation_window": correction_fallback,
        "out_of_domain_fraction_by_feature": diagnostics.get(
            "out_of_domain_fraction_by_feature", {}),
        "closure_overhead_fraction": closure_overhead,
        "repeated_particle_pair_fraction": repeated,
        "physical_run_pass": physical,
        "artifact_sha256": diagnostics.get("artifact_sha256"),
        "reference_provenance_sha256": diagnostics.get(
            "reference_provenance_sha256"),
        "runtime_seconds": float(diagnostics.get("runtime_seconds", 0.0)),
        "peak_rss_mib": float(diagnostics.get("peak_rss_mib", 0.0)),
    }


def summarize_case(items: list[dict], expected: int, sphere_reference: dict,
                   dem_reference: dict) -> dict:
    first = items[0]
    sphere = first["arm"] == "sphere_exact"
    fields = ["Tstar", *PK_COMPONENTS] + ([] if sphere else ["theta", "nematic_order"])
    means = {name: float(np.mean([item["means"][name] for item in items]))
             for name in fields}
    stds = {name: float(np.std([item["means"][name] for item in items], ddof=1))
            if len(items) > 1 else 0.0 for name in fields}
    cvs = {name: stds[name] / max(abs(means[name]), 1e-12) for name in fields}
    by_branch = defaultdict(list)
    for item in items:
        by_branch[item["initial_branch"]].append(item)
    branch_means = {
        branch: {name: float(np.mean([item["means"][name] for item in branch_items]))
                 for name in fields}
        for branch, branch_items in by_branch.items()}
    gaps = ({name: relative_gap(branch_means["cold"][name], branch_means["hot"][name])
             for name in fields}
            if set(branch_means) == {"cold", "hot"} else
            {name: FAIL_METRIC for name in fields})
    stationarity = (
        max(item["Tstar_relative_drift"] for item in items) <= 0.12
        and (sphere or max(item["theta_relative_drift"] for item in items) <= 0.10)
        and max(item["steady_energy_power_imbalance"] for item in items) <= 0.12)
    branch_pass = (gaps["Tstar"] <= 0.12
                   and max(gaps[name] for name in PK_COMPONENTS) <= 0.10
                   and (sphere or gaps["theta"] <= 0.10))
    precision = (cvs["Tstar"] <= 0.12
                 and max(cvs[name] for name in PK_COMPONENTS) <= 0.10
                 and (sphere or cvs["theta"] <= 0.10))
    fallback_fractions = [
        item["correction_fallback_fraction_in_evaluation_window"]
        for item in items]
    if first["arm"] != "corrected":
        support_class = "not_applicable"
    elif max(fallback_fractions) < THRESHOLDS[
            "correction_support_fraction_exclusive_maximum"]:
        support_class = "correction_supported"
    elif min(fallback_fractions) >= 0.99:
        support_class = "base_law_fallback"
    else:
        support_class = "mixed_support_base_law_fallback"
    feature_names = sorted({name for item in items
                            for name in item["out_of_domain_fraction_by_feature"]})
    record = {
        "coordinate_role": first["coordinate_role"], "arm": first["arm"],
        "aspect_ratio": first["aspect_ratio"], "alpha": first["alpha"],
        "dt": first["dt"], "rate_scale": first["rate_scale"],
        "n_expected_runs": expected, "n_valid_runs": len(items),
        "means": means, "standard_deviations": stds,
        "coefficients_of_variation": cvs, "initial_branch_means": branch_means,
        "initial_branch_relative_gaps": gaps,
        "maximum_Tstar_relative_drift": max(
            item["Tstar_relative_drift"] for item in items),
        "maximum_theta_relative_drift": (None if sphere else max(
            item["theta_relative_drift"] for item in items)),
        "maximum_steady_energy_power_imbalance": max(
            item["steady_energy_power_imbalance"] for item in items),
        "maximum_energy_ledger_increment_residual": max(
            item["energy_ledger_increment_residual"] for item in items),
        "maximum_correction_fallback_fraction": max(
            item["correction_fallback_fraction_in_evaluation_window"]
            for item in items),
        "mean_correction_fallback_fraction": float(np.mean(fallback_fractions)),
        "correction_support_class": support_class,
        "correction_support_pass": all(
            item["correction_support_pass"] for item in items),
        "adaptive_base_law_pass": all(
            item["adaptive_base_law_pass"] for item in items),
        "out_of_domain_fraction_by_feature_maximum": {
            name: max(item["out_of_domain_fraction_by_feature"].get(name, 0.0)
                      for item in items) for name in feature_names},
        "non_support_runtime_gate_reasons": sorted({
            reason for item in items
            for reason in item["non_support_runtime_gate_reasons"]}),
        "maximum_closure_overhead_fraction": max(
            item["closure_overhead_fraction"] for item in items),
        "maximum_repeated_particle_pair_fraction": max(
            item["repeated_particle_pair_fraction"] for item in items),
        "complete": len(items) == expected,
        "physical_run_pass": all(item["physical_run_pass"] for item in items),
        "stationarity_pass": bool(stationarity),
        "branch_convergence_pass": bool(branch_pass),
        "precision_pass": bool(precision),
    }
    if sphere:
        ref = sphere_reference[(first["alpha"],)]
        p = np.array([means[name] for name in PK_COMPONENTS])
        q = np.array([ref[name] for name in PK_COMPONENTS])
        record["sphere_benchmark"] = {
            "reference": {name: ref[name] for name in ("Tstar", *PK_COMPONENTS)},
            "Tstar_relative_error": abs(means["Tstar"] - ref["Tstar"])
            / abs(ref["Tstar"]),
            "kinetic_stress_relative_l2_error": relative_l2(p, q),
            "maximum_component_relative_error": max(
                abs(means[name] - ref[name]) / max(abs(ref[name]), 1e-12)
                for name in PK_COMPONENTS),
        }
        benchmark_pass = (
            record["sphere_benchmark"]["Tstar_relative_error"] <= 0.08
            and record["sphere_benchmark"]["kinetic_stress_relative_l2_error"] <= 0.05
            and record["sphere_benchmark"]["maximum_component_relative_error"] <= 0.10)
        record["sphere_benchmark_pass"] = bool(benchmark_pass)
    else:
        guide = interpolate_dem(dem_reference, first["aspect_ratio"], first["alpha"])
        record["dem_guidance_not_a_gate"] = {
            "reference": guide,
            "Tstar_relative_difference": abs(means["Tstar"] - guide["Tstar"])
            / max(abs(guide["Tstar"]), 1e-12),
            "theta_relative_difference": abs(means["theta"] - guide["theta"])
            / max(abs(guide["theta"]), 1e-12),
            "kinetic_stress_relative_l2_difference": relative_l2(
                np.array([means[name] for name in PK_COMPONENTS]),
                np.array([guide[name] for name in PK_COMPONENTS])),
        }
    record["internal_physics_pass"] = bool(
        record["complete"] and record["physical_run_pass"]
        and record["stationarity_pass"] and record["branch_convergence_pass"]
        and record["precision_pass"] and (not sphere or record["sphere_benchmark_pass"]))
    return record


def matching(cases: list[dict], **values) -> list[dict]:
    return [case for case in cases if all(case.get(key) == value
                                          for key, value in values.items())]


def convergence_comparisons(cases: list[dict], dimension: str) -> list[dict]:
    result = []
    if dimension == "dt":
        coarse = [case for case in cases if case["dt"] == 0.01]
        for a in coarse:
            candidates = matching(
                cases, arm=a["arm"], aspect_ratio=a["aspect_ratio"], alpha=a["alpha"],
                rate_scale=a["rate_scale"], dt=0.005)
            if not candidates:
                continue
            b = candidates[0]
            errors = {name: relative_gap(a["means"][name], b["means"][name])
                      for name in ("Tstar", *PK_COMPONENTS)}
            if a["arm"] != "sphere_exact":
                errors["theta"] = relative_gap(a["means"]["theta"], b["means"]["theta"])
            result.append({"aspect_ratio": a["aspect_ratio"], "alpha": a["alpha"],
                           "arm": a["arm"], "relative_differences": errors,
                           "pass": bool(errors["Tstar"] <= 0.08
                                        and max(errors[n] for n in PK_COMPONENTS) <= 0.05
                                        and (a["arm"] == "sphere_exact"
                                             or errors["theta"] <= 0.05))})
    else:
        low = [case for case in cases if case["rate_scale"] == 0.70]
        for a in low:
            candidates = matching(
                cases, arm=a["arm"], aspect_ratio=a["aspect_ratio"], alpha=a["alpha"],
                dt=a["dt"], rate_scale=1.0)
            if not candidates:
                continue
            b = candidates[0]
            errors = {name: relative_gap(a["means"][name], b["means"][name])
                      for name in ("Tstar", "theta", *PK_COMPONENTS)}
            result.append({"aspect_ratio": a["aspect_ratio"], "alpha": a["alpha"],
                           "arm": a["arm"], "relative_differences": errors,
                           "pass": bool(errors["Tstar"] <= 0.08
                                        and errors["theta"] <= 0.05
                                        and max(errors[n] for n in PK_COMPONENTS) <= 0.05)})
    return result


def paired_arm_comparisons(cases: list[dict]) -> list[dict]:
    """Report paired base/adaptive behavior without using DEM as a gate."""
    result = []
    corrected = [case for case in cases
                 if case["arm"] == "corrected" and case["dt"] == 0.005
                 and case["rate_scale"] == 1.0]
    for adaptive in corrected:
        candidates = matching(
            cases, arm="uncorrected",
            aspect_ratio=adaptive["aspect_ratio"], alpha=adaptive["alpha"],
            dt=adaptive["dt"], rate_scale=adaptive["rate_scale"])
        if not candidates:
            continue
        base = candidates[0]
        fields = ("Tstar", "theta", *PK_COMPONENTS, "nematic_order")
        differences = {name: relative_gap(
            adaptive["means"][name], base["means"][name]) for name in fields}
        result.append({
            "aspect_ratio": adaptive["aspect_ratio"],
            "alpha": adaptive["alpha"],
            "correction_support_class": adaptive["correction_support_class"],
            "relative_differences": differences,
            "internal_pair_pass": bool(
                adaptive["internal_physics_pass"]
                and base["internal_physics_pass"]),
            "interpretation": (
                "diagnostic_only_no_DEM_calibration_or_accuracy_gate"),
        })
    return result


def make_figure(cases: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (("Tstar", r"$T_{tr}/(m\dot\gamma^2d^2)$"),
              ("theta", r"$T_{tr}/T_{rot}$"),
              ("Pk_xy", r"$P^k_{xy}/nT_{tr}$"),
              ("Pk_xx", r"$P^k_{xx}/nT_{tr}$"))
    for axis, (field, label) in zip(axes.flat, panels):
        plotted = False
        groups = defaultdict(list)
        for case in cases:
            if case["dt"] != 0.005 or case["rate_scale"] != 1.0:
                continue
            if field not in case["means"]:
                continue
            groups[(case["arm"], case["aspect_ratio"])].append(case)
        for (arm, ar), selected in sorted(groups.items()):
            selected.sort(key=lambda item: item["alpha"])
            style = "--" if arm == "uncorrected" else "-"
            axis.plot([item["alpha"] for item in selected],
                      [item["means"][field] for item in selected], style,
                      marker="o", markersize=3, label=f"AR={ar:g} {arm}")
            plotted = True
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        if not plotted:
            axis.text(0.5, 0.5, "No valid results", ha="center", va="center",
                      transform=axis.transAxes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("USF cross-flow validation (DEM is guidance only)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sphere-reference", required=True)
    parser.add_argument("--dem-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args()
    manifest = rows(Path(args.manifest))
    modes = {row["mode"] for row in manifest}
    if len(modes) != 1:
        raise SystemExit("manifest must contain exactly one campaign mode")
    mode = modes.pop()
    expected = defaultdict(int)
    for row in manifest:
        key = (row["coordinate_role"], row["arm"], float(row["aspect_ratio"]),
               float(row["alpha"]), float(row["dt"]), float(row["rate_scale"]))
        expected[key] += 1
    loaded, failures = [], []
    for row in manifest:
        try:
            loaded.append(analyze_run(row))
        except Exception as error:  # a partial array must still get a QA summary
            failures.append({"task_id": int(row["task_id"]),
                             "output_prefix": row["output_prefix"],
                             "error": f"{type(error).__name__}: {error}"})
    grouped = defaultdict(list)
    for item in loaded:
        key = (item["coordinate_role"], item["arm"], item["aspect_ratio"],
               item["alpha"], item["dt"], item["rate_scale"])
        grouped[key].append(item)
    sphere_reference = load_reference(Path(args.sphere_reference), ("alpha",))
    dem_reference = load_reference(Path(args.dem_reference), ("AR", "alpha"))
    cases = [summarize_case(items, expected[key], sphere_reference, dem_reference)
             for key, items in sorted(grouped.items())]
    dt_checks = convergence_comparisons(cases, "dt")
    rate_checks = convergence_comparisons(cases, "rate")
    fine = [case for case in cases if case["dt"] == 0.005
            and case["rate_scale"] == 1.0]
    sphere_pass = bool(matching(fine, arm="sphere_exact")) and all(
        case["internal_physics_pass"] for case in matching(fine, arm="sphere_exact"))
    corrected_pass = bool(matching(fine, arm="corrected")) and all(
        case["internal_physics_pass"] for case in matching(fine, arm="corrected"))
    uncorrected_cases = matching(fine, arm="uncorrected")
    uncorrected_pass = (None if not uncorrected_cases else all(
        case["internal_physics_pass"] for case in uncorrected_cases))
    corrected_cases = matching(fine, arm="corrected")
    all_corrected_cases = matching(cases, arm="corrected")
    all_case_internal_pass = bool(cases) and all(
        case["internal_physics_pass"] for case in cases)
    adaptive_policy_pass = bool(all_corrected_cases) and all(
        case["adaptive_base_law_pass"] for case in all_corrected_cases)
    correction_coverage_pass = bool(corrected_cases) and all(
        case["correction_support_pass"] for case in corrected_cases)
    paired_comparisons = paired_arm_comparisons(cases)
    artifact_hashes = sorted({item["artifact_sha256"] for item in loaded
                              if item.get("artifact_sha256")})
    provenance_hashes = sorted({item["reference_provenance_sha256"] for item in loaded
                                if item.get("reference_provenance_sha256")})
    manifest_hashes = sorted({row.get("artifact_sha256") for row in manifest
                              if row.get("artifact_sha256")})
    provenance_consistent = bool(
        len(provenance_hashes) == 1 and len(manifest_hashes) == 1
        and artifact_hashes == manifest_hashes)
    complete = len(loaded) == len(manifest) and not failures and provenance_consistent
    if mode == "numerics":
        stage_pass = bool(complete and all_case_internal_pass
                          and sphere_pass and corrected_pass
                          and dt_checks and all(item["pass"] for item in dt_checks)
                          and rate_checks and all(item["pass"] for item in rate_checks))
    elif mode == "pilot":
        stage_pass = bool(
            complete and sphere_pass and corrected_pass and uncorrected_pass
            and len(paired_comparisons) == 9
            and all(item["internal_pair_pass"] for item in paired_comparisons))
    elif mode == "full":
        offgrid = matching(fine, coordinate_role="interpolation_holdout")
        stage_pass = bool(complete and sphere_pass and corrected_pass and offgrid
                          and all(case["internal_physics_pass"] for case in offgrid))
    elif mode == "response-validation":
        response_holdouts = matching(
            fine, coordinate_role="response_interpolation_holdout")
        stage_pass = bool(
            complete and sphere_pass and corrected_pass and uncorrected_pass
            and len(paired_comparisons) == 33
            and all(item["internal_pair_pass"] for item in paired_comparisons)
            and len(response_holdouts) == 24
            and all(item["internal_physics_pass"] for item in response_holdouts)
            and correction_coverage_pass)
    else:
        raise SystemExit(f"unknown campaign mode {mode}")
    summary = {
        "schema_version": "usf-crossflow-summary-v2", "protocol": PROTOCOL,
        "analysis_revision": ANALYSIS_REVISION,
        "correction_support_policy": SUPPORT_POLICY,
        "thresholds": THRESHOLDS,
        "mode": mode, "n_expected": len(manifest), "n_valid": len(loaded),
        "n_failures": len(failures), "failures": failures,
        "complete": complete, "artifact_sha256_values": artifact_hashes,
        "reference_provenance_sha256_values": provenance_hashes,
        "provenance_consistent": provenance_consistent,
        "sphere_benchmark_pass": sphere_pass,
        "corrected_internal_physics_pass": corrected_pass,
        "all_case_internal_physics_pass": all_case_internal_pass,
        "uncorrected_internal_physics_pass": uncorrected_pass,
        "adaptive_base_law_policy_pass": adaptive_policy_pass,
        "strict_correction_coverage_pass": correction_coverage_pass,
        "correction_support_case_counts": {
            label: sum(case["correction_support_class"] == label
                       for case in corrected_cases)
            for label in ("correction_supported", "base_law_fallback",
                          "mixed_support_base_law_fallback")},
        "paired_arm_comparisons": paired_comparisons,
        "time_step_convergence": dt_checks,
        "shear_rate_similarity": rate_checks,
        "stage_pass": stage_pass,
        "numerics_pass": stage_pass if mode == "numerics" else None,
        "pilot_pass": stage_pass if mode == "pilot" else None,
        "full_domain_pass": stage_pass if mode == "full" else None,
        "promotion_ready": bool(mode == "full" and stage_pass),
        # The evidence artifact remains non-deployable until a separate local
        # promotion step binds this complete full-domain summary into it.
        "deployment_ready": False,
        "comparison_policy": {
            "sphere": "accuracy_gate_against_independent_Boltzmann_DSMC",
            "rods": "DEM_guidance_only_not_a_gate_or_calibration_target",
            "correction_support": (
                "validated_angular_response_inside_support_and_fail_closed_"
                "unchanged_base_law_outside_support"),
            "primary_rod_stress": "kinetic",
            "rod_total_stress": "diagnostic_only_until_exact_branch_virial_exists",
        },
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_figure(cases, Path(args.figure))
    print(json.dumps({key: summary[key] for key in (
        "mode", "n_expected", "n_valid", "n_failures", "complete",
        "sphere_benchmark_pass", "corrected_internal_physics_pass",
        "all_case_internal_physics_pass",
        "adaptive_base_law_policy_pass", "strict_correction_coverage_pass",
        "correction_support_case_counts", "stage_pass", "promotion_ready", "deployment_ready")}, indent=2))


if __name__ == "__main__":
    main()
