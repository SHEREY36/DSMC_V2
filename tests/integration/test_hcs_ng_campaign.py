import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MAKE = ROOT / "DSMC_0D_v2/scripts/make_hcs_ng_manifest.py"


def _analysis_module():
    path = ROOT / "DSMC_0D_v2/scripts/analyze_hcs_ng_campaign.py"
    spec = importlib.util.spec_from_file_location("analyze_hcs_ng_campaign", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _test_artifact(tmp_path):
    surface = np.array([[alpha, theta, ar]
                        for alpha in (0.5, 0.8, 0.95, 1.0)
                        for theta in (0.025, 0.2, 1.0, 2.0)
                        for ar in (1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0)])
    beta = np.array([[alpha, theta, ar]
                     for alpha in (0.8, 0.95, 1.0)
                     for theta in (0.2, 1.0, 2.0)
                     for ar in (2.0, 3.0)])
    artifact = tmp_path / "closure.npz"
    np.savez_compressed(artifact, surface_coordinates=surface,
                        beta_coordinates=beta)
    return artifact


def make_manifest(tmp_path, mode, artifact, model_variant="baseline"):
    manifest = tmp_path / f"{mode}.csv"
    subprocess.run([
        sys.executable, str(MAKE), "--mode", mode, "--artifact", str(artifact),
        "--model-variant", model_variant,
        "--output", str(manifest), "--results", str(tmp_path / "results")],
        check=True, cwd=ROOT)
    with manifest.open(newline="") as handle:
        return manifest, list(csv.DictReader(handle))


def test_engineering_design_pairs_scaled_and_unscaled(tmp_path):
    _, rows = make_manifest(tmp_path, "engineering", _test_artifact(tmp_path))
    assert len(rows) == 120
    assert {row["arm"] for row in rows} == {"scaled", "unscaled", "dt_half"}
    assert len({(row["alpha"], row["aspect_ratio"]) for row in rows}) == 5
    assert {int(row["particles"]) for row in rows} == {10000}
    assert {row["protocol_version"] for row in rows} == {"hcs-ng-v4"}
    assert {row["model_variant"] for row in rows} == {"baseline"}
    assert {row["invariant_corrections"] for row in rows} == {"false"}
    assert {float(row["dt"]) for row in rows if row["arm"] == "scaled"} == {0.005}
    assert {float(row["dt"]) for row in rows if row["arm"] == "unscaled"} == {0.005}
    assert {float(row["dt"]) for row in rows if row["arm"] == "dt_half"} == {0.0025}


def test_domain_pilot_uses_every_artifact_alpha_ar_pair(tmp_path):
    _, rows = make_manifest(tmp_path, "domain-pilot", _test_artifact(tmp_path))
    assert len(rows) == 4 * 7 * 4
    assert {float(row["alpha"]) for row in rows} == {0.5, 0.8, 0.95, 1.0}
    assert {float(row["aspect_ratio"]) for row in rows} == {
        1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0}


def test_stability_design_brackets_roots_and_uses_common_dissipation_horizon(tmp_path):
    _, rows = make_manifest(tmp_path, "stability", _test_artifact(tmp_path))
    assert len(rows) == 148
    assert {float(row["initial_theta"]) for row in rows} == {0.025, 0.225, 1.5}
    assert all(float(row["initial_theta"]) in (
        (0.025, 1.5) if float(row["aspect_ratio"]) <= 1.35 else (0.225, 1.5))
        for row in rows)
    assert {float(row["dissipation_horizon"]) for row in rows
            if float(row["alpha"]) < 1.0} == {600.0}
    for row in rows:
        alpha = float(row["alpha"])
        if alpha < 1.0:
            assert np.isclose((1.0 - alpha**2) * float(row["tau_end"]), 600.0)
    assert {row["arm"] for row in rows} == {"scaled"}
    assert {float(row["dt"]) for row in rows if row["arm"] == "scaled"} == {0.005}


def test_stability_sentinel_repeats_long_horizon_at_half_dt(tmp_path):
    _, rows = make_manifest(
        tmp_path, "stability-sentinel", _test_artifact(tmp_path))
    assert len(rows) == 40
    assert len({(row["alpha"], row["aspect_ratio"]) for row in rows}) == 5
    assert {row["arm"] for row in rows} == {"scaled", "dt_half"}
    assert {float(row["initial_theta"]) for row in rows} == {0.025, 0.225, 1.5}
    assert {float(row["dissipation_horizon"]) for row in rows
            if float(row["alpha"]) < 1.0} == {600.0}
    assert {float(row["dt"]) for row in rows if row["arm"] == "dt_half"} == {0.0025}


def test_stability_requires_complete_passing_sentinel_on_same_bytes(tmp_path):
    artifact = _test_artifact(tmp_path)
    manifest, _ = make_manifest(tmp_path, "stability", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    pilot = tmp_path / "sentinel.json"
    command = [sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
               "--manifest", str(manifest), "--artifact", str(artifact),
               "--pilot-summary", str(pilot)]
    pilot.write_text(json.dumps({
        "mode": "stability-sentinel", "protocol_version": "hcs-ng-v4",
        "model_variant": "baseline", "invariant_corrections": False,
        "long_time_stability_campaign_pass": True,
        "artifact_sha256": digest,
        "n_tasks": 40, "n_completed_tasks": 40,
        "failed_tasks": [], "missing_tasks": [],
        "arm_dt": {"scaled": [0.005], "dt_half": [0.0025]},
    }))
    accepted = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert accepted.returncode == 0, accepted.stderr
    payload = json.loads(pilot.read_text())
    payload["long_time_stability_campaign_pass"] = False
    pilot.write_text(json.dumps(payload))
    blocked = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert blocked.returncode != 0


def test_sentinel_accepts_identical_passing_v3_engineering_gate(tmp_path):
    artifact = _test_artifact(tmp_path)
    manifest, _ = make_manifest(tmp_path, "stability-sentinel", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    summary = tmp_path / "engineering.json"
    cases = ((0.50, 1.35), (0.50, 3.0), (0.80, 2.0),
             (0.95, 2.0), (1.00, 3.0))
    summary.write_text(json.dumps({
        "mode": "engineering", "protocol_version": "hcs-ng-v3",
        "model_variant": "baseline", "invariant_corrections": False,
        "study_campaign_pass": True, "artifact_sha256": digest,
        "n_tasks": 120, "n_completed_tasks": 120,
        "arm_dt": {"scaled": [0.005], "unscaled": [0.005],
                   "dt_half": [0.0025]},
        "scaled_unscaled_equivalence": [
            {"control_arm": arm, "alpha": alpha, "aspect_ratio": ar,
             "pass": True}
            for arm in ("unscaled", "dt_half") for alpha, ar in cases
        ],
    }))
    result = subprocess.run([
        sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
        "--manifest", str(manifest), "--artifact", str(artifact),
        "--pilot-summary", str(summary),
    ], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_current_artifact_allows_engineering_but_blocks_domain_pilot(tmp_path):
    artifact = _test_artifact(tmp_path)
    engineering, _ = make_manifest(tmp_path, "engineering", artifact)
    allowed = subprocess.run([
        sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
        "--manifest", str(engineering), "--artifact", str(artifact),
        "--allow-engineering"], cwd=ROOT, text=True, capture_output=True)
    assert allowed.returncode == 0, allowed.stderr

    domain, _ = make_manifest(tmp_path, "domain-pilot", artifact)
    blocked = subprocess.run([
        sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
        "--manifest", str(domain), "--artifact", str(artifact)],
        cwd=ROOT, text=True, capture_output=True)
    assert blocked.returncode != 0
    assert "hcs-summary" in blocked.stderr


def test_angular_evidence_preflight_requires_matching_evidence_manifest(tmp_path):
    artifact = _test_artifact(tmp_path)
    with np.load(artifact, allow_pickle=False) as data:
        surface = np.asarray(data["surface_coordinates"])
    np.savez_compressed(artifact, surface_coordinates=surface,
                        beta_coordinates=surface)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest, _ = make_manifest(
        tmp_path, "engineering", artifact, model_variant="angular_evidence")
    command = [
        sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
        "--manifest", str(manifest), "--artifact", str(artifact),
        "--allow-engineering",
    ]
    missing = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert missing.returncode != 0
    (tmp_path / "manifest.json").write_text(json.dumps({
        "artifact_sha256": digest,
        "artifact_status": "evidence_only_not_deployable",
        "release_policy": "validated-angular-only-v1",
        "n_energy_release_nodes": 0,
    }))
    accepted = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert accepted.returncode == 0, accepted.stderr


def test_scientific_preflight_rejects_compact_gate_and_accepts_full_gate(tmp_path):
    surface = np.array([[alpha, theta, ar]
                        for alpha in (0.5, 0.8, 0.95, 1.0)
                        for theta in (0.2, 1.0, 2.0)
                        for ar in (1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0)])
    artifact = tmp_path / "full.npz"
    np.savez_compressed(artifact, surface_coordinates=surface,
                        beta_coordinates=surface)
    manifest, _ = make_manifest(tmp_path, "domain-pilot", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    summary = tmp_path / "hcs.json"
    command = [
        sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
        "--manifest", str(manifest), "--artifact", str(artifact),
        "--hcs-summary", str(summary),
    ]
    summary.write_text(json.dumps({"physics_gate_pass": True,
                                   "artifact_sha256": digest}))
    compact = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert compact.returncode != 0
    assert "full-domain" in compact.stderr
    summary.write_text(json.dumps({"physics_gate_pass": True,
                                   "full_domain_physics_gate_pass": True,
                                   "artifact_sha256": digest}))
    full = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert full.returncode == 0, full.stderr


def test_slurm_runner_strides_past_site_array_limit():
    text = (ROOT / "hpc/hcs_ng_array.slurm").read_text()
    assert "while (( INDEX < ROWS ))" in text
    assert "INDEX=$(( INDEX + STRIDE ))" in text
    analysis = (ROOT / "hpc/analyze_hcs_ng.slurm").read_text()
    assert "MPLBACKEND=Agg" in analysis
    assert "MPLCONFIGDIR" in analysis


def test_tail_fit_removes_radial_jacobian_and_obeys_count_gate(tmp_path):
    module = _analysis_module()
    edges = np.linspace(0.0, 8.0, 257)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.rint(2.0e6 * centers**2 * np.exp(-2.0 * centers)
                     * np.diff(edges)).astype(np.int64)
    histogram = tmp_path / "hist.npz"
    np.savez_compressed(histogram, c_edges=edges, c_counts=counts)
    item = {"histograms_file": str(histogram),
            "tail_counts": {"c": int(counts[centers >= 2.0].sum())},
            "tail_thresholds": {"c": 2.0}, "minimum_tail_count": 1000}
    fit = module.pooled_tail_fit([item], "c")
    assert fit["fit_ready"]
    assert np.isclose(fit["gamma"], 2.0, atol=0.03)
    blocked = dict(item, minimum_tail_count=10**12)
    assert not module.pooled_tail_fit([blocked], "c")["fit_ready"]


def test_marginal_panels_use_broad_restitution_comparison():
    module = _analysis_module()
    sweep = {(alpha, 2.0): [] for alpha in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)}
    tails = {(alpha, 2.0): [] for alpha in (0.5, 0.75, 0.95, 1.0)}
    assert module._representative_inelastic_alphas(sweep) == (0.5, 0.8, 0.95)
    assert module._representative_inelastic_alphas(tails) == (0.5, 0.75, 0.95)


def test_temperature_ratio_gate_is_reciprocal_invariant(tmp_path):
    module = _analysis_module()
    moments = tmp_path / "moments.csv"
    moments.write_text(
        "theta_tr_over_rot,theta_rot_over_tr\n"
        "0.25,4.0\n"
        "0.5,2.0\n"
    )
    series = module.load_series(moments)
    assert np.allclose(series[module.LOG_THETA], np.log([0.25, 0.5]))
    assert module.LOG_THETA in module.CONTROL_OBSERVABLES
    assert "theta_tr_over_rot" not in module.CONTROL_OBSERVABLES
    assert "theta_rot_over_tr" not in module.CONTROL_OBSERVABLES
    assert module.MAX_MAJORANT_VIOLATIONS_PER_ACCEPTED_PAIR == 1.0e-5


def test_stationarity_gate_detects_delayed_departure():
    module = _analysis_module()
    too_short = module.block_stationarity(
        np.zeros(module.MINIMUM_STATIONARITY_SAMPLES - 1), module.LOG_THETA)
    assert not too_short["pass"]
    assert too_short["reason"] == "too_few_late_samples"
    flat = np.zeros(40)
    delayed = flat.copy()
    delayed[30:] = np.linspace(0.0, 0.12, 10)
    assert module.block_stationarity(flat, module.LOG_THETA)["pass"]
    verdict = module.block_stationarity(delayed, module.LOG_THETA)
    assert not verdict["pass"]
    assert verdict["reason"] == "late_window_change_detected"


def test_full_domain_correction_preflight_fails_closed(tmp_path):
    surface = np.array([[0.8, 0.2, 2.0], [0.8, 1.0, 2.0],
                        [0.95, 0.2, 3.0], [0.95, 1.0, 3.0]])
    artifact = tmp_path / "artifact.npz"
    np.savez_compressed(artifact, surface_coordinates=surface,
                        beta_coordinates=surface[:-1])
    command = [sys.executable,
               str(ROOT / "hpc/require_full_domain_correction.py"),
               str(artifact)]
    blocked = subprocess.run(command, text=True, capture_output=True)
    assert blocked.returncode != 0
    assert "missing=1" in blocked.stderr
    np.savez_compressed(artifact, surface_coordinates=surface,
                        beta_coordinates=surface)
    assert subprocess.run(command).returncode == 0


def test_sweep_design_avoids_near_sphere_and_includes_elastic_control(tmp_path):
    _, rows = make_manifest(tmp_path, "sweep", _test_artifact(tmp_path))
    cases = {(float(row["alpha"]), float(row["aspect_ratio"])) for row in rows}
    assert len(cases) == 37 and len(rows) == 37 * 10
    assert min(ar for _, ar in cases) == 1.1
    assert {float(row["dt"]) for row in rows if row["arm"] == "scaled"} == {0.005}
    assert {alpha for alpha, ar in cases if ar == 2.0} == {
        0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0}
    assert {ar for alpha, ar in cases if alpha == 0.8} == {
        1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0}
    assert {int(row["particles"]) for row in rows} == {10000}
    assert {row["arm"] for row in rows} == {"scaled"}


def test_ng_analysis_runs_after_any_array_outcome():
    submit = (ROOT / "hpc/submit_hcs_ng_campaign.sh").read_text()
    assert '--dependency="afterany:$JOB"' in submit
    assert "--allow-missing" in (ROOT / "hpc/analyze_hcs_ng.slurm").read_text()


def test_sweep_requires_passing_long_time_stability_on_same_bytes(tmp_path):
    surface = np.array([[alpha, theta, ar]
                        for alpha in (0.5, 0.8, 0.95, 1.0)
                        for theta in (0.2, 1.0, 2.0)
                        for ar in (1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0)])
    artifact = tmp_path / "full.npz"
    # No beta surface: the sweep must not depend on the correction hull.
    np.savez_compressed(artifact, surface_coordinates=surface)
    manifest, _ = make_manifest(tmp_path, "sweep", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    pilot = tmp_path / "pilot.json"
    command = [sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
               "--manifest", str(manifest), "--artifact", str(artifact)]
    missing = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert missing.returncode != 0 and "pilot-summary" in missing.stderr
    for payload, ok in (
            ({"mode": "engineering", "protocol_version": "hcs-ng-v4",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": True,
              "artifact_sha256": digest}, False),
            ({"mode": "stability", "protocol_version": "hcs-ng-v3",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": True,
              "artifact_sha256": digest}, False),
            ({"mode": "stability", "protocol_version": "hcs-ng-v4",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": False,
              "n_tasks": 148, "n_completed_tasks": 148,
              "failed_tasks": [], "missing_tasks": [],
              "artifact_sha256": digest}, False),
            ({"mode": "stability", "protocol_version": "hcs-ng-v4",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": True,
              "artifact_sha256": "0" * 64}, False),
            ({"mode": "stability", "protocol_version": "hcs-ng-v4",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": True,
              "artifact_sha256": digest,
              "n_tasks": 148, "n_completed_tasks": 148,
              "failed_tasks": [], "missing_tasks": []}, True)):
        pilot.write_text(json.dumps(payload))
        result = subprocess.run(command + ["--pilot-summary", str(pilot)],
                                cwd=ROOT, text=True, capture_output=True)
        assert (result.returncode == 0) == ok, result.stderr


def test_tails_requires_complete_passing_sweep_on_same_bytes(tmp_path):
    artifact = _test_artifact(tmp_path)
    manifest, _ = make_manifest(tmp_path, "tails", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    summary = tmp_path / "sweep.json"
    command = [sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
               "--manifest", str(manifest), "--artifact", str(artifact),
               "--pilot-summary", str(summary)]
    payload = {
        "mode": "sweep", "protocol_version": "hcs-ng-v4",
        "model_variant": "baseline", "invariant_corrections": False,
        "study_campaign_pass": True, "scientific_outputs_released": True,
        "artifact_sha256": digest,
        "n_tasks": 370, "n_completed_tasks": 370,
        "failed_tasks": [], "missing_tasks": [],
    }
    summary.write_text(json.dumps(payload))
    accepted = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert accepted.returncode == 0, accepted.stderr
    payload["mode"] = "stability"
    summary.write_text(json.dumps(payload))
    blocked = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert blocked.returncode != 0


def test_ng_runner_uses_manifest_correction_variant():
    text = (ROOT / "DSMC_0D_v2/scripts/run_hcs_ng_task.py").read_text()
    assert 'row.get("invariant_corrections"' in text
    assert "invariant_corrections=invariant_corrections" in text
    assert "full_domain_baseline_candidate.yaml" in text
