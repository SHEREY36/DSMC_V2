#!/usr/bin/env python3
"""Cross-validate and calibrate the quadratic invariant response on exact USF CTC.

The local excitation campaign fixes the fourteen-dimensional functional form.
Direct USF replay may only rescale its linear/quadratic parts on training states;
an independent branch and the latest-time cold state decide release.  No DEM
quantity is read by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from coll_models_v2.correction_support import (
    energy_quantiles, tangent_quantiles,
)
from coll_models_v2.fit_coefficients import CORRECTION_PARAMETERS
from coll_models_v2.pipeline import (
    DIRECT_RESPONSE_PRECISION_CONTRACT,
    direct_response_precision_status,
    precision_status,
)
from coll_models_v2.projections import angular_quantiles
from dsmc_v2.artifact import VariationalClosure
from dsmc_v2_contracts import FEATURE_NAMES


PARAMETER_NAMES = tuple(name for _, name in CORRECTION_PARAMETERS)


def weighted(values: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, np.asarray(values), axes=(0, 0))


def percentile(values: list[float], q: float = 0.95) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--coefficient-rows", required=True)
    parser.add_argument("--direct-estimates", required=True)
    parser.add_argument("--output-coefficients", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    coefficient_path = Path(args.coefficient_rows)
    payload = json.loads(coefficient_path.read_text())
    if (payload.get("release_policy") != "quadratic-evidence-v1"
            or int(payload.get("response_order", 0)) != 2):
        raise SystemExit("input is not the validated quadratic local response")
    rows = payload["coefficient_rows"]
    lookup = {tuple(map(float, row["coordinates"])): row for row in rows}
    closure = VariationalClosure(args.base_artifact, corrections_enabled=True)
    with Path(args.base_artifact).open("rb") as handle:
        base_artifact_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    if set(lookup) != {tuple(q) for q in closure.coordinates.tolist()}:
        raise SystemExit("coefficient and artifact physical surfaces differ")

    observations = []
    legacy_precision_reclassifications = 0
    paths = sorted(Path(args.direct_estimates).glob("replay_*.json"))
    if len(paths) != 168:
        raise SystemExit(f"expected 168 direct replay estimates, found {len(paths)}")
    for path in paths:
        estimate = json.loads(path.read_text())
        if not estimate.get("qa", {}).get("sentinel_pass", False):
            raise SystemExit(f"direct estimate failed physics QA: {path}")
        direct_pass, direct_reasons = direct_response_precision_status(estimate)
        qa = estimate.get("qa", {})
        precision_contract = qa.get("precision_contract")
        if precision_contract is None:
            # Campaigns produced before direct-response QA was separated from
            # excitation QA may carry only the inapplicable lambda1
            # contribution reason. Re-evaluate the immutable numerical result
            # and retain an explicit audit count in the response report.
            legacy_pass, expected_legacy_reasons = precision_status(estimate)
            legacy_reasons = set(qa.get("continuation_reasons", []))
            if (bool(qa.get("precision_pass")) != legacy_pass
                    or list(qa.get("continuation_reasons", []))
                    != expected_legacy_reasons):
                raise SystemExit(
                    f"legacy direct precision record is inconsistent: {path}")
            if legacy_reasons - {"lambda1_contribution_precision"}:
                raise SystemExit(
                    f"legacy direct estimate has additional precision failures: {path}")
            if legacy_reasons:
                legacy_precision_reclassifications += 1
        elif precision_contract != DIRECT_RESPONSE_PRECISION_CONTRACT:
            raise SystemExit(
                f"unknown direct precision contract {precision_contract}: {path}")
        elif (bool(qa.get("precision_pass")) != direct_pass
              or list(qa.get("continuation_reasons", [])) != direct_reasons):
            raise SystemExit(f"direct precision record is inconsistent: {path}")
        if not direct_pass:
            raise SystemExit(
                f"direct estimate failed direct-response precision QA "
                f"({','.join(direct_reasons)}): {path}")
        provenance = estimate.get("direct_response_provenance", {})
        if provenance.get("base_artifact_sha256") != base_artifact_sha256:
            raise SystemExit(f"direct estimate/base-artifact hash mismatch: {path}")
        query = np.array([estimate["alpha"], estimate["theta"],
                          estimate["aspect_ratio"]], dtype=float)
        indices, physical_weights = closure._physical_vertex_weights(query)
        physical_rows = [lookup[tuple(closure.coordinates[index])]
                         for index in indices]
        nearest_row = physical_rows[int(np.argmax(physical_weights))]
        center = weighted([row["feature_center"] for row in physical_rows],
                          physical_weights)
        beta = weighted([row["beta"] for row in physical_rows], physical_weights)
        quadratic = weighted([row["beta_quadratic"] for row in physical_rows],
                             physical_weights)
        linear_mask = np.asarray(
            nearest_row.get("beta_evidence_deployed",
                            nearest_row["beta_deployed"]), dtype=bool)
        quadratic_mask = np.asarray(
            nearest_row.get("beta_quadratic_evidence_deployed",
                            nearest_row.get("beta_quadratic_deployed",
                                            linear_mask)), dtype=bool)
        base = np.r_[
            closure._weighted(closure.energy_parameters, indices, physical_weights),
            closure._weighted(closure.angular_parameters, indices, physical_weights),
        ]
        exact = np.asarray([
            estimate[section][name] for section, name in CORRECTION_PARAMETERS],
            dtype=float)
        standard_error = np.asarray([
            estimate.get("uncertainty", {}).get(name, {}).get(
                "standard_error", np.nan) for name in PARAMETER_NAMES], dtype=float)
        features = np.asarray([estimate["cell_features"][name]
                               for name in FEATURE_NAMES], dtype=float)
        delta = features - center
        observations.append({
            "path": str(path), "alpha": float(estimate["alpha"]),
            "query": query, "indices": indices, "physical_weights": physical_weights,
            "branch": provenance["initial_branch"],
            "window": int(provenance["window_index"]),
            "features": features, "base": base, "exact": exact,
            "standard_error": standard_error,
            "linear": (beta * linear_mask) @ delta,
            "quadratic": (quadratic * quadratic_mask) @ (delta * delta),
            "estimate": estimate,
        })

    alpha_nodes = (0.5, 0.8, 0.95)
    scales: dict[float, np.ndarray] = {}
    parameter_metrics: dict[str, dict] = {}
    for alpha in alpha_nodes:
        family = [item for item in observations if np.isclose(item["alpha"], alpha)]
        train = [item for item in family
                 if item["branch"] == "cold" and item["window"] < 3]
        holdout = [item for item in family
                   if not (item["branch"] == "cold" and item["window"] < 3)]
        fitted_scales = np.ones((len(PARAMETER_NAMES), 2))
        alpha_metrics = {}
        for parameter, name in enumerate(PARAMETER_NAMES):
            design = np.asarray([[item["linear"][parameter],
                                  item["quadratic"][parameter]] for item in train])
            target = np.asarray([item["exact"][parameter] - item["base"][parameter]
                                 for item in train])
            se = np.asarray([item["standard_error"][parameter] for item in train])
            finite = se[np.isfinite(se) & (se > 0.0)]
            floor = float(np.median(finite)) if len(finite) else 1.0
            weight = 1.0 / np.maximum(np.where(np.isfinite(se), se, floor), floor) ** 2
            information = design.T @ (weight[:, None] * design)
            ridge = max(1.0e-12, 0.01 * float(np.trace(information)) / 2.0)
            fitted = np.linalg.solve(
                information + ridge * np.eye(2),
                design.T @ (weight * target) + ridge * np.ones(2))
            fitted_scales[parameter] = np.clip(fitted, -4.0, 4.0)
            observed = np.asarray([
                item["exact"][parameter] - item["base"][parameter]
                for item in holdout])
            predicted = np.asarray([
                fitted_scales[parameter, 0] * item["linear"][parameter]
                + fitted_scales[parameter, 1] * item["quadratic"][parameter]
                for item in holdout])
            baseline_rmse = float(np.sqrt(np.mean(observed * observed)))
            corrected_rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
            response_scale = max(float(np.ptp(observed)),
                                 float(np.max(np.abs(observed))), 1.0e-12)
            holdout_se = np.asarray([
                item["standard_error"][parameter] for item in holdout], dtype=float)
            finite_holdout_se = holdout_se[
                np.isfinite(holdout_se) & (holdout_se > 0.0)]
            median_se = (float(np.median(finite_holdout_se))
                         if len(finite_holdout_se) else floor)
            resolved = bool(baseline_rmse > 2.0 * median_se)
            passed = bool(
                (corrected_rmse < 0.9 * baseline_rmse
                 and corrected_rmse / response_scale <= 0.50)
                if resolved else
                corrected_rmse <= baseline_rmse + 2.0 * median_se)
            alpha_metrics[name] = {
                "linear_scale": float(fitted_scales[parameter, 0]),
                "quadratic_scale": float(fitted_scales[parameter, 1]),
                "baseline_rmse": baseline_rmse,
                "corrected_rmse": corrected_rmse,
                "relative_rmse": corrected_rmse / response_scale,
                "resolved_response": resolved, "pass": passed,
                "n_training": len(train), "n_holdout": len(holdout),
            }
        scales[alpha] = fitted_scales
        parameter_metrics[f"{alpha:.2f}"] = alpha_metrics

    distribution_metrics = {}
    probability = np.linspace(0.0, 1.0, 129)
    group_release: dict[float, tuple[bool, bool]] = {}
    for alpha in alpha_nodes:
        holdout = [item for item in observations
                   if np.isclose(item["alpha"], alpha)
                   and not (item["branch"] == "cold" and item["window"] < 3)]
        energy_base, energy_corrected, angular_base, angular_corrected = [], [], [], []
        for item in holdout:
            correction = (scales[alpha][:, 0] * item["linear"]
                          + scales[alpha][:, 1] * item["quadratic"])
            estimate = item["estimate"]
            base_energy = dict(estimate["energy"])
            base_angular = dict(estimate["angular"])
            for index, (section, name) in enumerate(CORRECTION_PARAMETERS):
                target = base_energy if section == "energy" else base_angular
                target[name] = float(item["base"][index])
            predicted_energy, predicted_angular = dict(base_energy), dict(base_angular)
            for index, (section, name) in enumerate(CORRECTION_PARAMETERS):
                target = predicted_energy if section == "energy" else predicted_angular
                target[name] = float(item["base"][index] + correction[index])
            loss = float(estimate["energy"].get("mean_fractional_loss", 0.0))
            for z_in in (0.2, 0.5, 0.8):
                exact_q = energy_quantiles(estimate["energy"], z_in, loss, probability)
                base_q = energy_quantiles(base_energy, z_in, loss, probability)
                predicted_q = tangent_quantiles(
                    base_energy, predicted_energy, z_in, loss, probability)
                energy_base.append(float(np.mean(np.abs(base_q - exact_q))))
                energy_corrected.append(float(np.mean(np.abs(predicted_q - exact_q))))
            exact_a = angular_quantiles(np.asarray([
                estimate["angular"]["eta1"], estimate["angular"]["eta2"]]), probability)
            base_a = angular_quantiles(np.asarray([
                base_angular["eta1"], base_angular["eta2"]]), probability)
            corrected_a = angular_quantiles(np.asarray([
                predicted_angular["eta1"], predicted_angular["eta2"]]), probability)
            angular_base.append(float(np.mean(np.abs(base_a - exact_a))))
            angular_corrected.append(float(np.mean(np.abs(corrected_a - exact_a))))
        values = {
            "baseline_energy_w1_p95": percentile(energy_base),
            "corrected_energy_w1_p95": percentile(energy_corrected),
            "baseline_angular_w1_p95": percentile(angular_base),
            "corrected_angular_w1_p95": percentile(angular_corrected),
        }
        energy_parameters_pass = all(
            parameter_metrics[f"{alpha:.2f}"][name]["pass"]
            for name in PARAMETER_NAMES[:4])
        angular_parameters_pass = all(
            parameter_metrics[f"{alpha:.2f}"][name]["pass"]
            for name in PARAMETER_NAMES[4:])
        energy_pass = bool(energy_parameters_pass
                           and values["corrected_energy_w1_p95"]
                           < 0.9 * values["baseline_energy_w1_p95"])
        angular_pass = bool(angular_parameters_pass
                            and values["corrected_angular_w1_p95"]
                            < 0.9 * values["baseline_angular_w1_p95"])
        values.update(energy_parameter_pass=energy_parameters_pass,
                      angular_parameter_pass=angular_parameters_pass,
                      energy_release_pass=energy_pass,
                      angular_release_pass=angular_pass)
        distribution_metrics[f"{alpha:.2f}"] = values
        group_release[alpha] = energy_pass, angular_pass

    # Apply alpha-dependent response scales.  Direct states may expand only
    # the adjacent physical vertices and only for a response group that passed
    # its independent holdout branch/window.
    scale_alpha = np.asarray(alpha_nodes)
    for row in rows:
        alpha = float(row["coordinates"][0])
        scale = (np.ones((len(PARAMETER_NAMES), 2)) if alpha > scale_alpha[-1]
                 else np.asarray([
                     [np.interp(alpha, scale_alpha,
                                [scales[a][parameter, component]
                                 for a in alpha_nodes])
                      for component in range(2)]
                     for parameter in range(len(PARAMETER_NAMES))]))
        beta = np.asarray(row["beta"], dtype=float)
        quadratic = np.asarray(row["beta_quadratic"], dtype=float)
        beta *= scale[:, 0, None]
        quadratic *= scale[:, 1, None]
        row["beta"] = beta.tolist()
        row["beta_quadratic"] = quadratic.tolist()
        if alpha in group_release:
            deployed = np.asarray(row["beta_deployed"], dtype=bool)
            quadratic_deployed = np.asarray(
                row.get("beta_quadratic_deployed", deployed), dtype=bool)
            if not group_release[alpha][0]:
                deployed[:4] = False
                quadratic_deployed[:4] = False
            if not group_release[alpha][1]:
                deployed[4:] = False
                quadratic_deployed[4:] = False
            row["beta_deployed"] = deployed.tolist()
            row["beta_quadratic_deployed"] = quadratic_deployed.tolist()

    for item in observations:
        alpha = item["alpha"]
        if alpha not in group_release or not any(group_release[alpha]):
            continue
        for index in item["indices"]:
            row = lookup[tuple(closure.coordinates[index])]
            row_alpha = float(row["coordinates"][0])
            # A Delaunay simplex containing a query on an exact alpha plane
            # can include a numerically zero-weight vertex on the adjacent
            # plane. Direct evidence may widen support only on the plane that
            # actually supplied that evidence; interpolation between released
            # planes is exercised separately by untouched USF holdouts.
            if row_alpha not in group_release or not np.isclose(
                    row_alpha, alpha, atol=1.0e-12, rtol=0.0):
                continue
            deployed = np.asarray(row["beta_deployed"], dtype=bool)
            quadratic_deployed = np.asarray(
                row.get("beta_quadratic_deployed", deployed), dtype=bool)
            linear_evidence = np.asarray(
                row.get("beta_evidence_deployed", deployed), dtype=bool)
            quadratic_evidence = np.asarray(
                row.get("beta_quadratic_evidence_deployed",
                        quadratic_deployed), dtype=bool)
            if group_release[alpha][0]:
                # The local coefficients were fitted from the complete
                # excitation design even where its outer-amplitude gate held
                # deployment back.  A passed direct CTC holdout supplies the
                # missing evidence specifically on this USF-adjacent physical
                # vertex; the feature box below prevents a broader release.
                deployed[:4] = linear_evidence[:4]
                quadratic_deployed[:4] = quadratic_evidence[:4]
            if group_release[alpha][1]:
                deployed[4:] = linear_evidence[4:]
                quadratic_deployed[4:] = quadratic_evidence[4:]
            row["beta_deployed"] = deployed.tolist()
            row["beta_quadratic_deployed"] = quadratic_deployed.tolist()
            lower = np.asarray(row["feature_lower"], dtype=float)
            upper = np.asarray(row["feature_upper"], dtype=float)
            centre = np.asarray(row["feature_center"], dtype=float)
            pad = 0.05 * np.maximum(np.abs(item["features"] - centre), 1.0e-8)
            row["feature_lower"] = np.minimum(lower, item["features"] - pad).tolist()
            row["feature_upper"] = np.maximum(upper, item["features"] + pad).tolist()
            row["direct_usf_support"] = True

    response_group_decisions = {
        f"{alpha:.2f}": {
            "energy": ("released" if group_release[alpha][0]
                       else "suppressed_to_validated_base_law"),
            "angular": ("released" if group_release[alpha][1]
                        else "suppressed_to_validated_base_law"),
        }
        for alpha in alpha_nodes
    }
    payload.update({
        "release_policy": "quadratic-evidence-v1",
        "direct_response_contract": "cold_windows_0_2_train_hot_plus_late_holdout_v1",
        "direct_estimate_count": len(observations),
        "direct_precision_contract": DIRECT_RESPONSE_PRECISION_CONTRACT,
        "legacy_precision_reclassifications": legacy_precision_reclassifications,
        "direct_parameter_metrics": parameter_metrics,
        "direct_distribution_metrics": distribution_metrics,
        "direct_response_group_decisions": response_group_decisions,
        "dem_used_as_fit_target": False,
        "coefficient_rows": rows,
    })
    coefficient_rows_sha256 = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()).hexdigest()
    payload["direct_coefficient_rows_sha256"] = coefficient_rows_sha256
    complete_release = bool(all(
        group_release.get(alpha, (False, False)) == (True, True)
        for alpha in alpha_nodes))
    any_release = bool(any(any(value) for value in group_release.values()))
    decision_complete = set(group_release) == set(alpha_nodes)
    report = {
        "contract": payload["direct_response_contract"],
        "n_direct_estimates": len(observations),
        "direct_precision_contract": DIRECT_RESPONSE_PRECISION_CONTRACT,
        "legacy_precision_reclassifications": legacy_precision_reclassifications,
        "parameter_metrics": parameter_metrics,
        "distribution_metrics": distribution_metrics,
        "energy_release_alphas": [a for a, value in group_release.items() if value[0]],
        "angular_release_alphas": [a for a, value in group_release.items() if value[1]],
        "response_group_decisions": response_group_decisions,
        "complete_response_release": complete_release,
        # A failed response block is not deployed: its masks are cleared and
        # the already HCS-validated base law remains active. Building this
        # selective candidate is safe; deployment still requires the full
        # HCS and paired USF validation stages below this gate.
        "candidate_build_ready": bool(decision_complete and any_release),
        "coefficient_rows_sha256": coefficient_rows_sha256,
        "base_artifact_sha256": base_artifact_sha256,
        "dem_used_as_fit_target": False,
    }
    output_coefficients = Path(args.output_coefficients)
    output_report = Path(args.output_report)
    output_coefficients.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_coefficients.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["candidate_build_ready"]:
        raise SystemExit(
            "direct USF holdout did not release any response block; "
            "candidate build blocked")


if __name__ == "__main__":
    main()
