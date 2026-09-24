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


PROTOCOL_VERSION = "hcs-ng-v7"
ORIENTATION_INTEGRATOR = "symmetric_midpoint_v1"


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
                        help=("passing summary from the immediately preceding "
                              "engineering/stability stage"))
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
    expected_dt = ({"scaled": {0.0025}, "unscaled": {0.0025},
                    "dt_half": {0.00125}}
                   if mode in ("numerics-pilot", "engineering")
                   else {"scaled": {0.0025}, "dt_half": {0.00125}}
                   if mode == "stability-sentinel"
                   else {"scaled": {0.0025}}
                   if mode in ("stability", "sweep", "map", "tails") else None)
    if expected_dt is not None and arm_dt != expected_dt:
        raise SystemExit(
            f"{mode} manifest has wrong protocol-v7 step sizes: {arm_dt}")
    protocols = {row.get("protocol_version", "") for row in rows}
    if protocols != {PROTOCOL_VERSION}:
        raise SystemExit(
            f"manifest must use {PROTOCOL_VERSION}; found {sorted(protocols)}")
    variants = {row.get("model_variant", "") for row in rows}
    correction_flags = {row.get("invariant_corrections", "") for row in rows}
    manifest_hashes = {row.get("artifact_sha256", "") for row in rows}
    orientation_integrators = {
        row.get("orientation_integrator", "") for row in rows}
    if (len(variants) != 1 or len(correction_flags) != 1
            or len(manifest_hashes) != 1 or len(orientation_integrators) != 1):
        raise SystemExit("manifest mixes model provenance")
    if orientation_integrators != {ORIENTATION_INTEGRATOR}:
        raise SystemExit(
            "manifest must use the symmetric midpoint orientation integrator")
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
    if mode == "numerics-pilot":
        if not args.allow_engineering:
            raise SystemExit("numerics-pilot mode requires --allow-engineering")
    elif mode == "engineering":
        if not args.pilot_summary:
            raise SystemExit(
                "engineering mode requires --pilot-summary from a passing "
                "numerics pilot")
        pilot = json.loads(Path(args.pilot_summary).read_text())
        if pilot.get("mode") != "numerics-pilot":
            raise SystemExit("pilot summary is not a numerics-pilot summary")
        if pilot.get("protocol_version") != PROTOCOL_VERSION:
            raise SystemExit("numerics pilot predates the current HCS-NG protocol")
        if pilot.get("orientation_integrator") != ORIENTATION_INTEGRATOR:
            raise SystemExit("numerics pilot used a different orientation integrator")
        if pilot.get("model_variant") != model_variant:
            raise SystemExit("numerics pilot used a different model variant")
        if bool(pilot.get("invariant_corrections")) != corrections_enabled:
            raise SystemExit("numerics pilot used different correction routing")
        if pilot.get("artifact_sha256") != artifact_hash:
            raise SystemExit("numerics pilot did not use these artifact bytes")
        if not pilot.get("study_campaign_pass", False):
            raise SystemExit("numerics pilot study verdict has not passed")
        if pilot.get("n_tasks") != 24 or pilot.get("n_completed_tasks") != 24:
            raise SystemExit("numerics pilot is not the complete 24-task design")
        if pilot.get("failed_tasks") or pilot.get("missing_tasks"):
            raise SystemExit("numerics pilot contains failed or missing tasks")
        if pilot.get("arm_dt") != {
                "scaled": [0.0025], "unscaled": [0.0025],
                "dt_half": [0.00125]}:
            raise SystemExit("numerics pilot used the wrong protocol-v7 step sizes")
        controls = pilot.get("scaled_unscaled_equivalence", [])
        for control in ("unscaled", "dt_half"):
            passed = {(float(item["alpha"]), float(item["aspect_ratio"]))
                      for item in controls
                      if item.get("control_arm") == control and item.get("pass")}
            if passed != {(0.50, 1.35)}:
                raise SystemExit(
                    f"numerics pilot lacks the passing {control} control")
    elif mode == "stability-sentinel":
        if not args.pilot_summary:
            raise SystemExit(
                "stability-sentinel mode requires --pilot-summary from a passing "
                "engineering pilot")
        pilot = json.loads(Path(args.pilot_summary).read_text())
        if pilot.get("mode") != "engineering":
            raise SystemExit("pilot summary is not an engineering-mode summary")
        if pilot.get("protocol_version") != PROTOCOL_VERSION:
            raise SystemExit("engineering pilot is incompatible with HCS-NG v7")
        if pilot.get("orientation_integrator") != ORIENTATION_INTEGRATOR:
            raise SystemExit("engineering pilot used a different orientation integrator")
        if pilot.get("model_variant") != model_variant:
            raise SystemExit("engineering pilot used a different model variant")
        if bool(pilot.get("invariant_corrections")) != corrections_enabled:
            raise SystemExit("engineering pilot used different correction routing")
        if not pilot.get("study_campaign_pass", False):
            raise SystemExit("engineering pilot study verdict has not passed")
        if pilot.get("artifact_sha256") != artifact_hash:
            raise SystemExit("engineering pilot did not use these artifact bytes")
        if pilot.get("n_tasks") != 120:
            raise SystemExit("engineering pilot does not contain the 120-task protocol-v7 design")
        if pilot.get("arm_dt") != {
                "scaled": [0.0025], "unscaled": [0.0025],
                "dt_half": [0.00125]}:
            raise SystemExit("engineering pilot used the wrong protocol-v7 step sizes")
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
    elif mode == "stability":
        if not args.pilot_summary:
            raise SystemExit(
                "stability mode requires --pilot-summary from a passing "
                "long-time sentinel")
        pilot = json.loads(Path(args.pilot_summary).read_text())
        if pilot.get("mode") != "stability-sentinel":
            raise SystemExit("pilot summary is not a stability-sentinel summary")
        if pilot.get("protocol_version") != PROTOCOL_VERSION:
            raise SystemExit("stability sentinel predates the current HCS-NG protocol")
        if pilot.get("orientation_integrator") != ORIENTATION_INTEGRATOR:
            raise SystemExit("stability sentinel used a different orientation integrator")
        if pilot.get("model_variant") != model_variant:
            raise SystemExit("stability sentinel used a different model variant")
        if bool(pilot.get("invariant_corrections")) != corrections_enabled:
            raise SystemExit("stability sentinel used different correction routing")
        if not pilot.get("long_time_stability_campaign_pass", False):
            raise SystemExit("long-time sentinel gate has not passed")
        if pilot.get("artifact_sha256") != artifact_hash:
            raise SystemExit("stability sentinel did not use these artifact bytes")
        if pilot.get("n_tasks") != 80 or pilot.get("n_completed_tasks") != 80:
            raise SystemExit("stability sentinel is not the complete 80-task design")
        if pilot.get("failed_tasks") or pilot.get("missing_tasks"):
            raise SystemExit("stability sentinel contains failed or missing tasks")
        if pilot.get("arm_dt") != {"scaled": [0.0025], "dt_half": [0.00125]}:
            raise SystemExit("stability sentinel used the wrong step sizes")
    elif mode in ("sweep", "map"):
        if not args.pilot_summary:
            raise SystemExit(
                f"{mode} requires --pilot-summary from a passing long-time stability campaign")
        pilot = json.loads(Path(args.pilot_summary).read_text())
        if pilot.get("mode") != "stability":
            raise SystemExit("pilot summary is not a stability-mode summary")
        if pilot.get("protocol_version") != PROTOCOL_VERSION:
            raise SystemExit("stability pilot predates the current HCS-NG protocol")
        if pilot.get("orientation_integrator") != ORIENTATION_INTEGRATOR:
            raise SystemExit("stability pilot used a different orientation integrator")
        if pilot.get("model_variant") != model_variant:
            raise SystemExit("stability pilot used a different model variant")
        if bool(pilot.get("invariant_corrections")) != corrections_enabled:
            raise SystemExit("stability pilot used different correction routing")
        if not pilot.get("long_time_stability_campaign_pass", False):
            raise SystemExit("long-time two-sided HCS stability gate has not passed")
        if pilot.get("artifact_sha256") != artifact_hash:
            raise SystemExit("stability pilot did not use these artifact bytes")
        if pilot.get("n_tasks") != 148 or pilot.get("n_completed_tasks") != 148:
            raise SystemExit("stability pilot is not the complete 148-task design")
        if pilot.get("failed_tasks") or pilot.get("missing_tasks"):
            raise SystemExit("stability pilot contains failed or missing tasks")
    elif mode == "tails":
        if not args.pilot_summary:
            raise SystemExit(
                "tails requires --pilot-summary from the passing production sweep")
        pilot = json.loads(Path(args.pilot_summary).read_text())
        if pilot.get("mode") != "sweep":
            raise SystemExit("tail gate summary is not a sweep-mode summary")
        if pilot.get("protocol_version") != PROTOCOL_VERSION:
            raise SystemExit("production sweep predates the current HCS-NG protocol")
        if pilot.get("orientation_integrator") != ORIENTATION_INTEGRATOR:
            raise SystemExit("production sweep used a different orientation integrator")
        if pilot.get("model_variant") != model_variant:
            raise SystemExit("production sweep used a different model variant")
        if bool(pilot.get("invariant_corrections")) != corrections_enabled:
            raise SystemExit("production sweep used different correction routing")
        if not pilot.get("study_campaign_pass", False) \
                or not pilot.get("scientific_outputs_released", False):
            raise SystemExit("production sweep scientific verdict has not passed")
        if pilot.get("artifact_sha256") != artifact_hash:
            raise SystemExit("production sweep did not use these artifact bytes")
        if pilot.get("n_tasks") != 370 or pilot.get("n_completed_tasks") != 370:
            raise SystemExit("production sweep is not the complete 370-task design")
        if pilot.get("failed_tasks") or pilot.get("missing_tasks"):
            raise SystemExit("production sweep contains failed or missing tasks")
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
            # Validate the actual declared starts. The runtime additionally
            # stops inside the hull if an evolving trajectory approaches it.
            for theta in (float(row.get("initial_theta") or 1.0),):
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
