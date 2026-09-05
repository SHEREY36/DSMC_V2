"""Grid discovery and variational node-estimation orchestration."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from dsmc_v2_contracts import FEATURE_NAMES, load_run

from .estimate import estimate_node
from .weights import DEFAULT_OFFSETS


def discover_runs(root: str | Path) -> list[Path]:
    return sorted(metadata.parent for metadata in Path(root).rglob("metadata_v2.json")
                  if (metadata.parent / "_SUCCESS").is_file())


def group_runs(paths) -> dict[tuple[float, float, float, int], list[Path]]:
    grouped = defaultdict(list)
    for path in paths:
        run = load_run(path)
        key = (float(run.metadata["alpha"]), float(run.metadata["theta"]),
               float(run.metadata["aspect_ratio"]), int(run.metadata.get("ensemble_id", 0)))
        grouped[key].append(Path(path))
    return dict(grouped)


def precision_status(result: dict) -> tuple[bool, list[str]]:
    if "energy" not in result:
        # Read-only schema-2.1 QA adapter. Clock/loss discrepancies stay audit
        # only, exactly as they did in the released legacy pipeline.
        reasons = []
        if not result["qa"].get("vss_representable", True) and result["theta"] == 1.0:
            reasons.append("vss_unrepresentable")
        if result["alpha"] < 1.0:
            fc = result["quantities"]["F_C"]
            if fc["ci_low"] is None or 0.5 * (fc["ci_high"] - fc["ci_low"]) > 0.01 * abs(fc["estimate"]):
                reasons.append("F_C_precision")
            if not result["qa"].get("score_tail_pass", False):
                reasons.append("score_tail_instability")
            for name in FEATURE_NAMES:
                row = result["quantities"][f"beta_ctc_{name}"]
                if row["ci_low"] is None:
                    reasons.append(f"beta_ctc_{name}_missing")
        return not reasons, reasons
    reasons = []
    qa = result["qa"]
    for key in ("propensity_pass", "proposal_balance_pass", "ess_pass",
                "energy_projection_pass", "angular_projection_pass",
                "model_form_pass", "memory_diagnostic_pass",
                "incoming_partition_pass", "elastic_pass"):
        if not qa.get(key, False):
            reasons.append(key.removesuffix("_pass"))
    for name in ("p_exch", "reset_mean", "lambda1", "lambda2", "lambda3",
                 "eta1", "eta2"):
        interval = result.get("uncertainty", {}).get(name)
        if interval is None:
            reasons.append(f"{name}_precision_missing")
    # Excitation continuation is driven by the contribution uncertainty, not
    # a relative coefficient error that diverges at a true zero coefficient.
    if int(result.get("ensemble_id", 0)) != 0:
        x = result["proposal_features"]
        lambda_se = result.get("uncertainty", {}).get("lambda1", {}).get("standard_error")
        if lambda_se is None or max(abs(value) for value in x.values()) * 1.96 * lambda_se > 0.005:
            reasons.append("lambda1_contribution_precision")
    return not reasons, reasons


def estimate_grid(runs_root: str | Path, output_directory: str | Path,
                  bl=None, n_bootstrap: int = 200,
                  propensity_offsets: int | None = DEFAULT_OFFSETS) -> list[dict]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    grouped = group_runs(discover_runs(runs_root))
    results = []
    for key, paths in sorted(grouped.items()):
        result = estimate_node(paths, bl, n_bootstrap=n_bootstrap,
                               propensity_offsets=propensity_offsets)
        passed, reasons = precision_status(result)
        result["qa"].update(precision_pass=passed, continuation_reasons=reasons)
        results.append(result)
        tag = (f"alpha_{key[0]:.3f}_theta_{key[1]:.3f}_AR_{key[2]:.3f}_"
               f"ensemble_{key[3]:03d}.json")
        (output / tag).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    long_fields = ["alpha", "theta", "aspect_ratio", "ensemble_id", "quantity",
                   "estimate", "standard_error", "ci_low", "ci_high"]
    with (output / "closure_coefficients_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for result in results:
            for section in ("energy", "angular"):
                for name, value in result[section].items():
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    uncertainty = result.get("uncertainty", {}).get(name, {})
                    writer.writerow({
                        "alpha": result["alpha"], "theta": result["theta"],
                        "aspect_ratio": result["aspect_ratio"],
                        "ensemble_id": result["ensemble_id"],
                        "quantity": name, "estimate": value,
                        "standard_error": uncertainty.get("standard_error"),
                        "ci_low": uncertainty.get("ci_low"),
                        "ci_high": uncertainty.get("ci_high"),
                    })
    with (output / "qa_summary.csv").open("w", newline="") as handle:
        fields = ["alpha", "theta", "aspect_ratio", "ensemble_id", "n_attempts",
                  "n_outcomes", "precision_pass", "sentinel_pass", "ess_fraction",
                  "propensity_pass", "proposal_balance_pass", "model_form_pass",
                  "continuation_reasons"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "alpha": result["alpha"], "theta": result["theta"],
                "aspect_ratio": result["aspect_ratio"], "ensemble_id": result["ensemble_id"],
                "n_attempts": result["n_attempts"], "n_outcomes": result["n_outcomes"],
                "precision_pass": result["qa"]["precision_pass"],
                "sentinel_pass": result["qa"]["sentinel_pass"],
                "ess_fraction": result["qa"]["ess_fraction"],
                "propensity_pass": result["qa"]["propensity_pass"],
                "proposal_balance_pass": result["qa"]["proposal_balance_pass"],
                "model_form_pass": result["qa"]["model_form_pass"],
                "continuation_reasons": ";".join(result["qa"]["continuation_reasons"]),
            })
    return results
