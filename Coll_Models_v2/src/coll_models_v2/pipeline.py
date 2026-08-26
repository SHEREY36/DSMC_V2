"""Grid discovery and node-wise estimation orchestration."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from dsmc_v2_contracts import FEATURE_NAMES, load_run

from .estimate import estimate_node
from .legacy_bl import LegacyBL


def discover_runs(root: str | Path) -> list[Path]:
    runs = []
    for metadata in Path(root).rglob("metadata_v2.json"):
        directory = metadata.parent
        if (directory / "_SUCCESS").is_file():
            runs.append(directory)
    return sorted(runs)


def group_runs(paths) -> dict[tuple[float, float, float], list[Path]]:
    grouped = defaultdict(list)
    for path in paths:
        run = load_run(path)
        key = (float(run.metadata["alpha"]), float(run.metadata["theta"]),
               float(run.metadata["aspect_ratio"]))
        grouped[key].append(Path(path))
    return dict(grouped)


def _precision_status(result: dict) -> tuple[bool, list[str]]:
    reasons = []
    sigma = result["quantities"]["sigma_ctc"]
    if sigma["ci_low"] is None or 0.5 * (sigma["ci_high"] - sigma["ci_low"]) > 0.01 * abs(sigma["estimate"]):
        reasons.append("cross_section_qa_precision")
    if not result["qa"]["cross_section_pass"]:
        reasons.append("cross_section_polynomial_disagreement")
    if not result["qa"]["vss_representable"] and result["theta"] == 1.0:
        reasons.append("vss_unrepresentable")
    if result["alpha"] < 1.0:
        f0 = result["quantities"]["F0"]
        if f0["ci_low"] is None or 0.5 * (f0["ci_high"] - f0["ci_low"]) > 0.01 * abs(f0["estimate"]):
            reasons.append("F0_precision")
        if not result["qa"]["total_loss_compatibility_pass"]:
            reasons.append("preserved_BL_total_loss_mismatch")
        if not result["qa"]["score_tail_pass"]:
            reasons.append("score_tail_instability")
        for index, feature in enumerate(FEATURE_NAMES):
            row = result["quantities"][f"beta_{feature}"]
            if row["ci_low"] is None:
                reasons.append(f"beta_{feature}_missing")
                continue
            half = 0.5 * (row["ci_high"] - row["ci_low"])
            threshold = max(0.05, 0.15 * abs(row["estimate"])) if index < 12 \
                else max(0.10, 0.25 * abs(row["estimate"]))
            if half > threshold:
                reasons.append(f"beta_{feature}_precision")
    if result["theta"] == 1.0:
        b2 = result["quantities"]["B2"]
        if b2["ci_low"] is None or 0.5 * (b2["ci_high"] - b2["ci_low"]) > 0.01 * abs(b2["estimate"]):
            reasons.append("B2_precision")
    return not reasons, reasons


def estimate_grid(runs_root: str | Path, output_directory: str | Path,
                  bl: LegacyBL, n_bootstrap: int = 2000) -> list[dict]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    grouped = group_runs(discover_runs(runs_root))
    results = []
    for key, paths in sorted(grouped.items()):
        result = estimate_node(paths, bl, n_bootstrap=n_bootstrap)
        passed, reasons = _precision_status(result)
        result["qa"].update(precision_pass=passed, continuation_reasons=reasons)
        results.append(result)
        tag = f"alpha_{key[0]:.3f}_theta_{key[1]:.3f}_AR_{key[2]:.3f}.json"
        (output / tag).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    long_fields = ["alpha", "theta", "aspect_ratio", "quantity", "estimate",
                   "standard_error", "ci_low", "ci_high"]
    with (output / "closure_coefficients_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for result in results:
            for name, row in result["quantities"].items():
                writer.writerow({"alpha": result["alpha"], "theta": result["theta"],
                    "aspect_ratio": result["aspect_ratio"], "quantity": name, **row})
    with (output / "qa_summary.csv").open("w", newline="") as handle:
        fields = ["alpha", "theta", "aspect_ratio", "n_attempts", "n_outcomes",
                  "precision_pass", "vss_representable", "continuation_reasons"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for result in results:
            writer.writerow({
                "alpha": result["alpha"], "theta": result["theta"],
                "aspect_ratio": result["aspect_ratio"], "n_attempts": result["n_attempts"],
                "n_outcomes": result["n_outcomes"],
                "precision_pass": result["qa"]["precision_pass"],
                "vss_representable": result["qa"]["vss_representable"],
                "continuation_reasons": ";".join(result["qa"]["continuation_reasons"]),
            })
    return results
