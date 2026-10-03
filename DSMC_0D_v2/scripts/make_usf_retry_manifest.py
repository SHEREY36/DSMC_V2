#!/usr/bin/env python3
"""Build a compact Slurm retry manifest from missing or incomplete USF tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


OUTPUT_SUFFIXES = {
    ".txt": 5,
    "_pressure.txt": 14,
    "_orientation.txt": 8,
    "_energy.txt": 6,
}


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def table_status(path: Path, width: int, evaluation_start_tau: float) -> str | None:
    """Return a precise reason when a sampled output is incomplete or corrupt."""
    n_rows = 0
    n_evaluation = 0
    previous_tau = -math.inf
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != width:
                return f"invalid_schema:{path}:line_{line_number}"
            try:
                values = [float(value) for value in fields]
            except ValueError:
                return f"nonnumeric_row:{path}:line_{line_number}"
            if not all(math.isfinite(value) for value in values):
                return f"nonfinite_row:{path}:line_{line_number}"
            tau = values[1]
            if tau + 1.0e-12 < previous_tau:
                return f"nonmonotonic_tau:{path}:line_{line_number}"
            previous_tau = tau
            n_rows += 1
            n_evaluation += tau >= evaluation_start_tau
    if n_rows == 0:
        return f"missing_data_rows:{path}"
    if n_evaluation < 20:
        return f"insufficient_evaluation_samples:{path}:{n_evaluation}"
    return None


def task_status(row: dict[str, str]) -> tuple[bool, str]:
    prefix = row["output_prefix"]
    evaluation_start_tau = float(row["evaluation_start_tau"])
    for suffix, width in OUTPUT_SUFFIXES.items():
        path = Path(prefix + suffix)
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"missing_or_empty:{path}"
        reason = table_status(path, width, evaluation_start_tau)
        if reason is not None:
            return False, reason
    diagnostics_path = Path(prefix + ".json")
    if not diagnostics_path.is_file() or diagnostics_path.stat().st_size == 0:
        return False, f"missing_or_empty:{diagnostics_path}"
    try:
        diagnostics = json.loads(diagnostics_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid_json:{diagnostics_path}:{error}"
    validation = diagnostics.get("validation_case") or {}
    required = {
        "artifact_sha256": row["artifact_sha256"],
        "termination_reason": "collision_target",
        "particles": int(row["particles"]),
        "usf_study_protocol": "usf-crossflow-v1",
    }
    for key, expected in required.items():
        if diagnostics.get(key) != expected:
            return False, f"diagnostic_mismatch:{key}"
    validation_required = {
        "mode": row["mode"],
        "coordinate_role": row["coordinate_role"],
        "arm": row["arm"],
        "initial_branch": row["initial_branch"],
        "replicate": int(row["replicate"]),
        "seed": int(row["seed"]),
        "particles": int(row["particles"]),
    }
    for key, expected in validation_required.items():
        if validation.get(key) != expected:
            return False, f"validation_case_mismatch:{key}"
    validation_floats = {
        "alpha": row["alpha"],
        "aspect_ratio": row["aspect_ratio"],
        "shear_rate": row["shear_rate"],
        "rate_scale": row["rate_scale"],
        "dt": row["dt"],
        "volume_fraction": row["volume_fraction"],
        "tau_end": row["tau_end"],
        "evaluation_start_tau": row["evaluation_start_tau"],
    }
    for key, expected in validation_floats.items():
        actual = validation.get(key)
        if actual is None or not math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12):
            return False, f"validation_case_mismatch:{key}"
    source_task_id = validation.get(
        "source_task_id", validation.get("task_id", -1))
    if int(source_task_id) != int(row["task_id"]):
        return False, "validation_case_mismatch:source_task_id"
    support_policy = diagnostics.get("usf_study_support_policy")
    if support_policy not in (None, "adaptive_base_law_v1"):
        return False, "diagnostic_mismatch:usf_study_support_policy"
    return True, "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    source = Path(args.manifest)
    fieldnames, rows = read_manifest(source)
    if not rows or "task_id" not in fieldnames:
        raise SystemExit("source manifest is empty or lacks task_id")
    if [int(row["task_id"]) for row in rows] != list(range(len(rows))):
        raise SystemExit("source manifest task IDs are not contiguous")

    missing = []
    for row in rows:
        complete, reason = task_status(row)
        if not complete:
            retry = dict(row)
            retry["source_task_id"] = row["task_id"]
            retry["task_id"] = str(len(missing))
            missing.append((retry, reason))

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    retry_fields = fieldnames + ([] if "source_task_id" in fieldnames
                                 else ["source_task_id"])
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=retry_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row, _ in missing)

    report = {
        "source_manifest": str(source),
        "retry_manifest": str(destination),
        "n_expected": len(rows),
        "n_complete": len(rows) - len(missing),
        "n_retry": len(missing),
        "retry_source_task_ids": [
            int(row["source_task_id"]) for row, _ in missing],
        "retry_reasons": [
            {"source_task_id": int(row["source_task_id"]), "reason": reason}
            for row, reason in missing],
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    compact = {
        "source_manifest": report["source_manifest"],
        "retry_manifest": report["retry_manifest"],
        "n_expected": report["n_expected"],
        "n_complete": report["n_complete"],
        "n_retry": report["n_retry"],
        "first_retry_source_task_id": (
            report["retry_source_task_ids"][0]
            if report["retry_source_task_ids"] else None),
        "last_retry_source_task_id": (
            report["retry_source_task_ids"][-1]
            if report["retry_source_task_ids"] else None),
        "detailed_report": args.report,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
