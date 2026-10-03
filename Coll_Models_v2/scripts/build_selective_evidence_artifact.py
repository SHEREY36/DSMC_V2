#!/usr/bin/env python3
"""Install a validated correction surface without rebuilding sampler tables.

This utility is intentionally limited to evidence artifacts.  It reuses the
frozen schema-2.4 baseline and sampler tables and replaces only the small
natural-parameter response arrays.  The input coefficient file must declare
the ``validated-angular-only-v1`` policy, which holds every energy row at zero.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np

from dsmc_v2.artifact import VariationalClosure
from dsmc_v2_contracts import FEATURE_NAMES

from coll_models_v2.fit_coefficients import CORRECTION_PARAMETER_NAMES


RELEASE_POLICY = "validated-angular-only-v1"
REPLACED_ARRAYS = {
    "beta_coordinates": lambda rows: np.asarray(
        [row["coordinates"] for row in rows], dtype=float),
    "beta": lambda rows: np.asarray([row["beta"] for row in rows], dtype=float),
    "beta_se": lambda rows: np.asarray(
        [row["beta_se"] for row in rows], dtype=float),
    "beta_deployed": lambda rows: np.asarray(
        [row["beta_deployed"] for row in rows], dtype=bool),
    "beta_feature_center": lambda rows: np.asarray(
        [row["feature_center"] for row in rows], dtype=float),
    "beta_feature_lower": lambda rows: np.asarray(
        [row["feature_lower"] for row in rows], dtype=float),
    "beta_feature_upper": lambda rows: np.asarray(
        [row["feature_upper"] for row in rows], dtype=float),
    "beta_trust_amplitude": lambda rows: np.asarray(
        [row["trust_amplitude"] for row in rows], dtype=float),
    "correction_parameter_names": lambda rows: np.asarray(
        CORRECTION_PARAMETER_NAMES),
    "correction_trust_amplitude": lambda rows: np.asarray(0.25),
    "evidence_release_policy": lambda rows: np.asarray(RELEASE_POLICY),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def validate_rows(payload: dict, base_artifact: Path) -> list[dict]:
    if payload.get("schema") != "correction-coefficient-surface-v1" \
            or payload.get("release_policy") != RELEASE_POLICY:
        raise ValueError("coefficient file is not an angular-only evidence release")
    rows = payload.get("coefficient_rows", [])
    if len(rows) != 144:
        raise ValueError(f"expected 144 physical coefficient nodes, found {len(rows)}")
    deployed = np.asarray([row["beta_deployed"] for row in rows], dtype=bool)
    if deployed.shape != (144, len(CORRECTION_PARAMETER_NAMES), len(FEATURE_NAMES)):
        raise ValueError(f"unexpected deployed-mask shape {deployed.shape}")
    if np.any(deployed[:, :4]):
        raise ValueError("evidence policy must hold all energy corrections at zero")
    if not np.any(deployed[:, 4:]):
        raise ValueError("evidence policy released no angular response")
    with np.load(base_artifact, allow_pickle=False) as base:
        if str(base["schema_version"]) != "2.4.0":
            raise ValueError("base artifact must use schema 2.4.0")
        surface = {tuple(item) for item in np.asarray(
            base["surface_coordinates"], dtype=float).tolist()}
    coordinates = {tuple(row["coordinates"]) for row in rows}
    if coordinates != surface:
        raise ValueError("coefficient coordinates do not cover the base surface exactly")
    return rows


def copy_with_replacements(source: Path, destination: Path,
                           replacements: dict[str, np.ndarray]) -> None:
    expected = {f"{name}.npy" for name in replacements}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "r") as old, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED,
            allowZip64=True) as new:
        present = {item.filename for item in old.infolist()}
        # Both arrays were introduced after the first schema-2.4 candidates;
        # adding them is backward-compatible with the runtime reader.
        additions = {"evidence_release_policy.npy", "beta_trust_amplitude.npy"}
        missing = expected - present - additions
        if missing:
            raise ValueError(f"base artifact lacks replaceable arrays {sorted(missing)}")
        for item in old.infolist():
            if item.filename in expected:
                continue
            with old.open(item, "r") as source_handle, new.open(
                    item, "w", force_zip64=True) as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
        for name, value in replacements.items():
            new.writestr(f"{name}.npy", npy_bytes(np.asarray(value)))
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--coefficient-rows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_artifact = Path(args.base_artifact)
    base_manifest_path = Path(args.base_manifest)
    coefficient_path = Path(args.coefficient_rows)
    output = Path(args.output)
    payload = json.loads(coefficient_path.read_text())
    rows = validate_rows(payload, base_artifact)
    replacements = {name: factory(rows)
                    for name, factory in REPLACED_ARRAYS.items()}

    artifact_path = output / "closure_v2.npz"
    copy_with_replacements(base_artifact, artifact_path, replacements)
    # Loading through the production reader verifies all tensor dimensions and
    # the exact runtime contract, including the energy-sensitivity dependency.
    closure = VariationalClosure(artifact_path, corrections_enabled=True)

    deployed = replacements["beta_deployed"]
    trust = replacements["beta_trust_amplitude"]
    manifest = json.loads(base_manifest_path.read_text())
    manifest.update({
        "file": artifact_path.name,
        "artifact_status": "evidence_only_not_deployable",
        "release_policy": RELEASE_POLICY,
        "release_decision": (
            "energy corrections held back; independently validated angular "
            "rows retained for diagnostic HCS/USF comparison"),
        "base_artifact": str(base_artifact),
        "base_artifact_sha256": sha256(base_artifact),
        "coefficient_rows": str(coefficient_path),
        "coefficient_source_digest": payload["source_digest"],
        "coefficient_file_sha256": sha256(coefficient_path),
        "n_nodes": int(payload["n_source_nodes"]),
        "n_coefficient_nodes": len(rows),
        "n_energy_release_nodes": int(np.count_nonzero(
            np.any(deployed[:, :4], axis=(1, 2)))),
        "n_angular_release_nodes": int(np.count_nonzero(
            np.any(deployed[:, 4:], axis=(1, 2)))),
        "n_fully_suppressed_nodes": int(np.count_nonzero(
            ~np.any(deployed, axis=(1, 2)))),
        "correction_trust_amplitudes": sorted(set(trust.tolist())),
        "n_expanded_correction_nodes": int(np.count_nonzero(trust > 0.25)),
        "correction_bounds": [0.0, 0.0],
        "correction_digest": hashlib.sha256(json.dumps(
            rows, sort_keys=True).encode()).hexdigest(),
        "coefficient_validation_relative_rmse_max": max(
            max(row["validation_relative_rmse"][name]
                for name in ("eta1", "eta2")
                if row["validation_relative_rmse"][name] is not None)
            for row in rows),
        "heldout_support_validation": [
            {"coordinates": row["coordinates"],
             "angular_release": row["angular_release"],
             "trust_amplitude": row["trust_amplitude"],
             **row["heldout_support_validation"]}
            for row in rows],
        "runtime_load_verified": True,
        "runtime_coefficient_shape": list(closure.beta.shape),
        "artifact_sha256": sha256(artifact_path),
    })
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "artifact_status", "release_policy", "n_coefficient_nodes",
        "n_energy_release_nodes", "n_angular_release_nodes",
        "n_fully_suppressed_nodes", "correction_trust_amplitudes",
        "artifact_sha256")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
