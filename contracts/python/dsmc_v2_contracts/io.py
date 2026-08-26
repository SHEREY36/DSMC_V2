"""Versioned binary I/O and run finalization for HS_CTC_v2."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, pair_score_kernel


SCHEMA_VERSION = "2.0.0"
ATTEMPT_REAL_NAMES = (
    *(f"c1_{a}" for a in "xyz"), *(f"c2_{a}" for a in "xyz"),
    *(f"omega1_{a}" for a in "xyz"), *(f"omega2_{a}" for a in "xyz"),
    *(f"u1_{a}" for a in "xyz"), *(f"u2_{a}" for a in "xyz"),
    *(f"impact_{a}" for a in "xyz"),
)
OUTCOME_REAL_NAMES = (
    *(f"c1_pre_{a}" for a in "xyz"), *(f"c2_pre_{a}" for a in "xyz"),
    *(f"omega1_pre_{a}" for a in "xyz"), *(f"omega2_pre_{a}" for a in "xyz"),
    *(f"u1_pre_{a}" for a in "xyz"), *(f"u2_pre_{a}" for a in "xyz"),
    *(f"c1_post_{a}" for a in "xyz"), *(f"c2_post_{a}" for a in "xyz"),
    *(f"omega1_post_{a}" for a in "xyz"), *(f"omega2_post_{a}" for a in "xyz"),
    *(f"u1_post_{a}" for a in "xyz"), *(f"u2_post_{a}" for a in "xyz"),
    *(f"impact_{a}" for a in "xyz"),
    *(f"contact_normal_{a}" for a in "xyz"),
    *(f"centerline_{a}" for a in "xyz"),
    "contact_lambda", "contact_mu", "et_elastic", "er_elastic",
    "et_inelastic", "er1_inelastic", "er2_inelastic", "e_initial",
    "delta_tr", "delta_rot", "delta_total", "elastic_rel_error",
    "b_contact", "b_out",
    *(f"ghat_pre_{a}" for a in "xyz"), *(f"ghat_post_{a}" for a in "xyz"),
)

ATTEMPT_DTYPE = np.dtype([
    ("event_id", "<i8"), ("attempt_index", "<i8"), ("block_id", "<i8"),
    ("hit", "<i4"), ("reserved", "<i4"),
    ("values", "<f8", (len(ATTEMPT_REAL_NAMES),)),
], align=False)
OUTCOME_DTYPE = np.dtype([
    ("event_id", "<i8"), ("attempt_index", "<i8"), ("block_id", "<i8"),
    ("n_contact", "<i4"), ("reserved", "<i4"),
    ("values", "<f8", (len(OUTCOME_REAL_NAMES),)),
], align=False)
AI = {name: i for i, name in enumerate(ATTEMPT_REAL_NAMES)}
OI = {name: i for i, name in enumerate(OUTCOME_REAL_NAMES)}


@dataclass(frozen=True)
class RunDataV2:
    directory: Path
    metadata: dict
    attempts: np.ndarray
    outcomes: np.ndarray


def _read_exact(path: Path, dtype: np.dtype) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size == 0 or size % dtype.itemsize:
        raise ValueError(f"{path} is empty, truncated, or has the wrong schema")
    return np.memmap(path, dtype=dtype, mode="r", shape=(size // dtype.itemsize,))


def load_run(directory: str | Path) -> RunDataV2:
    directory = Path(directory)
    metadata_path = directory / "metadata_v2.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {metadata.get('schema_version')!r}")
    if metadata.get("byte_order") != "little":
        raise ValueError("v2 records must be little-endian")
    if int(metadata.get("attempt_record_bytes", -1)) != ATTEMPT_DTYPE.itemsize \
            or int(metadata.get("outcome_record_bytes", -1)) != OUTCOME_DTYPE.itemsize:
        raise ValueError("metadata record sizes do not match schema v2")
    return RunDataV2(
        directory,
        metadata,
        _read_exact(directory / "attempts_v2.bin", ATTEMPT_DTYPE),
        _read_exact(directory / "outcomes_v2.bin", OUTCOME_DTYPE),
    )


def _vec(values: np.ndarray, index: dict[str, int], prefix: str) -> np.ndarray:
    return np.column_stack([values[:, index[f"{prefix}_{a}"]] for a in "xyz"])


def attempt_scores(run: RunDataV2) -> np.ndarray:
    values = np.asarray(run.attempts["values"])
    return pair_score_kernel(
        _vec(values, AI, "c1"), _vec(values, AI, "c2"),
        _vec(values, AI, "omega1"), _vec(values, AI, "omega2"),
        _vec(values, AI, "u1"), _vec(values, AI, "u2"),
        float(run.metadata["velocity_scale"]), float(run.metadata["omega_scale"]),
    )


def validate_run(run: RunDataV2, elastic_tolerance: float = 5.0e-3) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    attempts, outcomes = np.asarray(run.attempts), np.asarray(run.outcomes)
    akeys = np.column_stack((attempts["event_id"], attempts["attempt_index"]))
    okeys = np.column_stack((outcomes["event_id"], outcomes["attempt_index"]))
    if len(np.unique(akeys, axis=0)) != len(akeys):
        errors.append("attempt keys are not unique")
    if len(np.unique(okeys, axis=0)) != len(okeys):
        errors.append("outcome keys are not unique")
    hit_keys = akeys[attempts["hit"] == 1]
    if len(hit_keys) != len(outcomes) or set(map(tuple, hit_keys)) != set(map(tuple, okeys)):
        errors.append("hit attempts and accepted outcomes are not one-to-one")
    expected = int(run.metadata["nsamples"])
    if len(outcomes) != expected:
        errors.append(f"outcome count {len(outcomes)} != nsamples {expected}")
    if not np.isfinite(attempts["values"]).all() or not np.isfinite(outcomes["values"]).all():
        errors.append("binary records contain NaN or Inf")
    if np.any((attempts["hit"] != 0) & (attempts["hit"] != 1)):
        errors.append("attempt hit flags are not binary")
    unique_events, hits_per_event = np.unique(attempts["event_id"][attempts["hit"] == 1], return_counts=True)
    if len(unique_events) != expected or np.any(hits_per_event != 1):
        errors.append("every requested event must terminate in exactly one hit")
    if np.any(attempts["block_id"] < 0) or np.any(attempts["block_id"] >= 128):
        errors.append("attempt block IDs must lie in 0..127")
    if np.any(outcomes["n_contact"] < 1):
        errors.append("accepted outcomes must have at least one contact")

    av, ov = attempts["values"], outcomes["values"]
    for label, values, index, prefixes in (
        ("attempt", av, AI, ("u1", "u2")),
        ("outcome", ov, OI, ("u1_pre", "u2_pre", "u1_post", "u2_post", "ghat_pre", "ghat_post")),
    ):
        for prefix in prefixes:
            error = float(np.max(np.abs(np.linalg.norm(_vec(values, index, prefix), axis=1) - 1.0)))
            if error > 2.0e-6:
                errors.append(f"{label} {prefix} unit-vector error {error:.3e}")
    for prefix in ("omega1", "omega2"):
        w, u = _vec(av, AI, prefix), _vec(av, AI, prefix.replace("omega", "u"))
        error = float(np.max(np.abs(np.einsum("ni,ni->n", w, u)) / np.maximum(1.0, np.linalg.norm(w, axis=1))))
        if error > 2.0e-6:
            errors.append(f"attempt {prefix}-axis perpendicularity error {error:.3e}")
    for suffix in ("pre", "post"):
        for particle in ("1", "2"):
            w = _vec(ov, OI, f"omega{particle}_{suffix}")
            u = _vec(ov, OI, f"u{particle}_{suffix}")
            error = float(np.max(np.abs(np.einsum("ni,ni->n", w, u)) /
                                 np.maximum(1.0, np.linalg.norm(w, axis=1))))
            if error > 2.0e-6:
                errors.append(f"outcome omega{particle}_{suffix}-axis perpendicularity error {error:.3e}")
    for prefix in ("contact_normal", "centerline"):
        error = float(np.max(np.abs(np.linalg.norm(_vec(ov, OI, prefix), axis=1) - 1.0)))
        if error > 2.0e-6:
            errors.append(f"outcome {prefix} unit-vector error {error:.3e}")

    delta_tr = ov[:, OI["delta_tr"]]
    delta_rot = ov[:, OI["delta_rot"]]
    delta = ov[:, OI["delta_total"]]
    identity_error = float(np.max(np.abs(delta - delta_tr - delta_rot) /
                                  np.maximum(1.0, np.abs(ov[:, OI["e_initial"]]))))
    reference_error = float(np.max(np.maximum(
        np.abs(delta_tr - (ov[:, OI["et_elastic"]] - ov[:, OI["et_inelastic"]])),
        np.abs(delta_rot - (ov[:, OI["er_elastic"]] - ov[:, OI["er1_inelastic"]]
                            - ov[:, OI["er2_inelastic"]]))) /
        np.maximum(1.0, np.abs(ov[:, OI["e_initial"]]))))
    identity_error = max(identity_error, reference_error)
    if identity_error > 5.0e-12:
        errors.append(f"dissipation identity error {identity_error:.3e}")
    elastic_error = float(np.max(np.abs(ov[:, OI["elastic_rel_error"]])))
    if elastic_error > elastic_tolerance:
        errors.append(f"elastic replay error {elastic_error:.3e} exceeds {elastic_tolerance:.3e}")
    alpha = float(run.metadata["alpha"])
    if alpha < 1.0 and float(np.sum(delta)) <= 0.0:
        errors.append("aggregate dissipative denominator is not positive")
    energy_fields = ["et_elastic", "er_elastic", "et_inelastic", "er1_inelastic", "er2_inelastic", "e_initial"]
    if np.any(np.column_stack([ov[:, OI[name]] for name in energy_fields]) < -1.0e-12):
        errors.append("energy components must be nonnegative")
    negative_fraction = float(np.mean(delta < 0.0))
    if negative_fraction:
        warnings.append(f"{negative_fraction:.3%} of events have negative total loss")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "n_attempts": int(len(attempts)),
        "n_hits": int(np.sum(attempts["hit"])),
        "n_outcomes": int(len(outcomes)),
        "elastic_error_max": elastic_error,
        "dissipation_identity_error_max": identity_error,
        "negative_loss_fraction": negative_fraction,
    }


def finalize_run(directory: str | Path, require_pass: bool = True) -> dict:
    """Validate raw files and materialize exact 128-block sufficient statistics."""
    run = load_run(directory)
    qa = validate_run(run)
    if require_pass and qa["status"] != "pass":
        raise ValueError(json.dumps(qa, indent=2))
    scores = attempt_scores(run)
    attempts, outcomes = np.asarray(run.attempts), np.asarray(run.outcomes)
    hit = attempts["hit"].astype(bool)
    outcome_by_key = {
        (int(row["event_id"]), int(row["attempt_index"])): row for row in outcomes
    }
    delta = np.zeros(len(attempts))
    delta_tr = np.zeros(len(attempts))
    e_initial = np.zeros(len(attempts))
    angle_b2 = np.zeros(len(attempts))
    for i in np.flatnonzero(hit):
        row = outcome_by_key[(int(attempts[i]["event_id"]), int(attempts[i]["attempt_index"]))]
        values = row["values"]
        delta[i] = values[OI["delta_total"]]
        delta_tr[i] = values[OI["delta_tr"]]
        e_initial[i] = values[OI["e_initial"]]
        cosine = float(np.dot(
            np.array([values[OI[f"ghat_pre_{a}"]] for a in "xyz"]),
            np.array([values[OI[f"ghat_post_{a}"]] for a in "xyz"]),
        ))
        angle_b2[i] = 1.0 - 0.5 * (3.0 * cosine * cosine - 1.0)
    path = Path(directory) / "attempt_blocks_v2.csv"
    fields = ["block_id", "n_try", "n_hit", "sum_delta", "sum_delta_tr", "sum_e_initial", "sum_b2"]
    for name in FEATURE_NAMES:
        fields.extend((f"sum_try_K_{name}", f"sum_hit_K_{name}",
                       f"sum_delta_K_{name}", f"sum_delta_tr_K_{name}",
                       f"sum_e_initial_K_{name}"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block in range(128):
            mask = attempts["block_id"] == block
            hmask = mask & hit
            row = {
                "block_id": block, "n_try": int(np.sum(mask)), "n_hit": int(np.sum(hmask)),
                "sum_delta": float(np.sum(delta[mask])), "sum_delta_tr": float(np.sum(delta_tr[mask])),
                "sum_e_initial": float(np.sum(e_initial[mask])), "sum_b2": float(np.sum(angle_b2[mask])),
            }
            for j, name in enumerate(FEATURE_NAMES):
                row[f"sum_try_K_{name}"] = float(np.sum(scores[mask, j]))
                row[f"sum_hit_K_{name}"] = float(np.sum(scores[hmask, j]))
                row[f"sum_delta_K_{name}"] = float(np.sum(delta[mask] * scores[mask, j]))
                row[f"sum_delta_tr_K_{name}"] = float(np.sum(delta_tr[mask] * scores[mask, j]))
                row[f"sum_e_initial_K_{name}"] = float(np.sum(e_initial[mask] * scores[mask, j]))
            writer.writerow(row)
    metadata = dict(run.metadata)
    metadata.update(n_attempts=len(attempts), n_outcomes=len(outcomes), finalized=True)
    (Path(directory) / "metadata_v2.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (Path(directory) / "qa_v2.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n")
    schema = {
        "schema_version": SCHEMA_VERSION, "byte_order": "little",
        "attempt": {"key": ["seed", "event_id", "attempt_index"],
                    "record_bytes": ATTEMPT_DTYPE.itemsize,
                    "real_fields": list(ATTEMPT_REAL_NAMES)},
        "outcome": {"key": ["seed", "event_id", "attempt_index"],
                    "record_bytes": OUTCOME_DTYPE.itemsize,
                    "real_fields": list(OUTCOME_REAL_NAMES)},
    }
    (Path(directory) / "schema_v2.json").write_text(json.dumps(schema, indent=2) + "\n")
    if qa["status"] == "pass":
        (Path(directory) / "_SUCCESS").write_text("schema=2.0.0\n")
    return qa
