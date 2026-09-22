#!/usr/bin/env python3
"""Fail closed when a scientific HCS-NG design outruns artifact evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


PROTOCOL_VERSION = "hcs-ng-v3"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--hcs-summary")
    parser.add_argument("--pilot-summary",
                        help="passing engineering-pilot summary; gates the sweep")
    parser.add_argument("--allow-engineering", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.manifest).open(newline="")))
    if not rows:
        raise SystemExit("empty non-Gaussian manifest")
    modes = {row["mode"] for row in rows}
    if len(modes) != 1:
        raise SystemExit("manifest mixes campaign modes")
    mode = modes.pop()
    arm_dt = {}
    for row in rows:
        arm_dt.setdefault(row["arm"], set()).add(float(row["dt"]))
    expected_dt = ({"scaled": {0.005}, "unscaled": {0.005},
                    "dt_half": {0.0025}}
                   if mode == "engineering"
                   else {"scaled": {0.005}} if mode == "sweep" else None)
    if expected_dt is not None and arm_dt != expected_dt:
        raise SystemExit(
            f"{mode} manifest has wrong protocol-v3 step sizes: {arm_dt}")
    protocols = {row.get("protocol_version", "") for row in rows}
    if protocols != {PROTOCOL_VERSION}:
        raise SystemExit(
            f"manifest must use {PROTOCOL_VERSION}; found {sorted(protocols)}")
    variants = {row.get("model_variant", "") for row in rows}
    correction_flags = {row.get("invariant_corrections", "") for row in rows}
    manifest_hashes = {row.get("artifact_sha256", "") for row in rows}
    if len(variants) != 1 or len(correction_flags) != 1 or len(manifest_hashes) != 1:
        raise SystemExit("manifest mixes model provenance")
    model_variant = variants.pop()
    corrections_enabled = correction_flags.pop() == "true"
    artifact = Path(args.artifact)
    artifact_hash = digest(artifact)
    if manifest_hashes != {artifact_hash}:
        raise SystemExit("manifest was not generated for these artifact bytes")

    artifact_manifest_path = artifact.with_name("manifest.json")
    artifact_manifest = (json.loads(artifact_manifest_path.read_text())
                         if artifact_manifest_path.is_file() else {})
    if model_variant == "angular_evidence":
        if not corrections_enabled:
            raise SystemExit("angular-evidence campaign must enable invariant corrections")
        if artifact_manifest.get("artifact_sha256") != artifact_hash:
            raise SystemExit("angular-evidence manifest hash does not match the NPZ")
        if artifact_manifest.get("artifact_status") != "evidence_only_not_deployable":
            raise SystemExit("angular-evidence artifact has an unexpected status")
        if artifact_manifest.get("release_policy") != "validated-angular-only-v1":
            raise SystemExit("artifact is not the validated angular-only evidence model")
        if int(artifact_manifest.get("n_energy_release_nodes", -1)) != 0:
            raise SystemExit("angular-evidence artifact unexpectedly releases energy response")
    elif model_variant == "baseline":
        if corrections_enabled:
            raise SystemExit("baseline campaign must disable invariant corrections")
    elif model_variant != "sphere_exact":
        raise SystemExit(f"unknown model variant {model_variant!r}")
    if mode == "engineering":
        if not args.allow_engineering:
            raise SystemExit("engineering mode requires --allow-engineering")
    elif mode == "sweep":
        # The sweep is an HCS-only campaign on one frozen model variant. It is
        # gated on the engineering pilot of the same bytes: stationarity,
        # scaled/unscaled consistency, half-step convergence, bounded Monte
        # Carlo resolution, and zero runtime repairs.
        if not args.pilot_summary:
            raise SystemExit("sweep requires --pilot-summary from a passing engineering pilot")
        pilot = json.loads(Path(args.pilot_summary).read_text())
        if pilot.get("mode") != "engineering":
            raise SystemExit("pilot summary is not an engineering-mode summary")
        if pilot.get("protocol_version") != PROTOCOL_VERSION:
            raise SystemExit("engineering pilot predates the current HCS-NG protocol")
        if pilot.get("model_variant") != model_variant:
            raise SystemExit("engineering pilot used a different model variant")
        if bool(pilot.get("invariant_corrections")) != corrections_enabled:
            raise SystemExit("engineering pilot used different correction routing")
        if not pilot.get("study_campaign_pass", False):
            raise SystemExit("engineering pilot study verdict has not passed")
        if pilot.get("artifact_sha256") != artifact_hash:
            raise SystemExit("engineering pilot did not use these artifact bytes")
        if pilot.get("n_tasks") != 120:
            raise SystemExit("engineering pilot does not contain the 120-task protocol-v3 design")
        if pilot.get("arm_dt") != {
                "scaled": [0.005], "unscaled": [0.005], "dt_half": [0.0025]}:
            raise SystemExit("engineering pilot used the wrong protocol-v3 step sizes")
        if pilot.get("n_completed_tasks") != pilot.get("n_tasks"):
            raise SystemExit("engineering pilot is incomplete")
        controls = pilot.get("scaled_unscaled_equivalence", [])
        expected_cases = {(0.50, 1.35), (0.50, 3.0), (0.80, 2.0),
                          (0.95, 2.0), (1.00, 3.0)}
        for control in ("unscaled", "dt_half"):
            passed = {(float(item["alpha"]), float(item["aspect_ratio"]))
                      for item in controls
                      if item.get("control_arm") == control and item.get("pass")
                      and "alpha" in item and "aspect_ratio" in item}
            if passed != expected_cases:
                raise SystemExit(
                    f"engineering pilot lacks the five passing {control} controls")
    elif mode != "sphere-controls":
        if not args.hcs_summary:
            raise SystemExit("scientific campaign requires --hcs-summary")
        hcs = json.loads(Path(args.hcs_summary).read_text())
        if not hcs.get("full_domain_physics_gate_pass", False):
            raise SystemExit("expanded full-domain HCS physics gate has not passed")
        if hcs.get("artifact_sha256") != artifact_hash:
            raise SystemExit("expanded HCS gate did not use these artifact bytes")

    if mode != "sphere-controls":
        # The sampler must support every inelastic start.  When the angular
        # evidence response is enabled, its full correction surface must
        # support the same coordinates as well.
        with np.load(artifact, allow_pickle=False) as data:
            points = np.asarray(data["surface_coordinates"], dtype=float)
            correction_points = (np.asarray(data["beta_coordinates"], dtype=float)
                                 if corrections_enabled else None)
        if len(points) < 4:
            raise SystemExit("artifact has no interpolable sampler surface")
        hull = Delaunay(points)
        correction_hull = (Delaunay(correction_points)
                           if correction_points is not None else None)
        unsupported = []
        for row in rows:
            if float(row["alpha"]) >= 1.0:
                continue   # exact elastic block; never queries the artifact
            # Trajectories start at theta=1 and move toward theta_H; the
            # runtime still fails closed if one actually leaves the hull.
            for theta in (1.0,):
                query = [float(row["alpha"]), theta, float(row["aspect_ratio"])]
                if hull.find_simplex(query) < 0 and not np.any(np.all(
                        np.isclose(points, query, atol=1e-12), axis=1)):
                    unsupported.append(query); break
                if correction_hull is not None \
                        and correction_hull.find_simplex(query) < 0 \
                        and not np.any(np.all(
                            np.isclose(correction_points, query, atol=1e-12), axis=1)):
                    unsupported.append(query); break
        if unsupported:
            preview = unsupported[:5]
            raise SystemExit(f"artifact sampler/correction hull misses campaign cases {preview}"
                             f" ({len(unsupported)} rows unsupported)")
    print(f"HCS-NG prerequisites pass for {mode}: {len(rows)} tasks; "
          f"protocol={PROTOCOL_VERSION}; model={model_variant}; "
          f"corrections={corrections_enabled}; artifact={artifact_hash}")


if __name__ == "__main__":
    main()
