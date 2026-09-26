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
    assert {row["protocol_version"] for row in rows} == {"hcs-ng-v8"}
    assert {row["orientation_integrator"] for row in rows} == {
        "symmetric_midpoint_v1"}
    assert {row["model_variant"] for row in rows} == {"baseline"}
    assert {row["invariant_corrections"] for row in rows} == {"false"}
    assert {float(row["dt"]) for row in rows if row["arm"] == "scaled"} == {0.0025}
    assert {float(row["dt"]) for row in rows if row["arm"] == "unscaled"} == {0.0025}
    assert {float(row["dt"]) for row in rows if row["arm"] == "dt_half"} == {0.00125}


def test_numerics_pilot_is_the_failed_v6_coordinate_only(tmp_path):
    _, rows = make_manifest(
        tmp_path, "numerics-pilot", _test_artifact(tmp_path))
    assert len(rows) == 24
    assert {(float(row["alpha"]), float(row["aspect_ratio"]))
            for row in rows} == {(0.5, 1.35)}
    assert len({int(row["seed"]) for row in rows}) == 8
    assert {row["arm"] for row in rows} == {"scaled", "unscaled", "dt_half"}
    assert {row["orientation_integrator"] for row in rows} == {
        "symmetric_midpoint_v1"}


def test_engineering_requires_complete_passing_numerics_pilot(tmp_path):
    artifact = _test_artifact(tmp_path)
    manifest, _ = make_manifest(tmp_path, "engineering", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    pilot = tmp_path / "numerics.json"
    command = [
        sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
        "--manifest", str(manifest), "--artifact", str(artifact),
        "--pilot-summary", str(pilot),
    ]
    pilot.write_text(json.dumps({
        "mode": "numerics-pilot", "protocol_version": "hcs-ng-v8",
        "orientation_integrator": "symmetric_midpoint_v1",
        "model_variant": "baseline", "invariant_corrections": False,
        "study_campaign_pass": True, "artifact_sha256": digest,
        "n_tasks": 24, "n_completed_tasks": 24,
        "failed_tasks": [], "missing_tasks": [],
        "arm_dt": {"scaled": [0.0025], "unscaled": [0.0025],
                   "dt_half": [0.00125]},
        "scaled_unscaled_equivalence": [
            {"control_arm": arm, "alpha": 0.5, "aspect_ratio": 1.35,
             "pass": True}
            for arm in ("unscaled", "dt_half")
        ],
    }))
    accepted = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert accepted.returncode == 0, accepted.stderr
    payload = json.loads(pilot.read_text())
    payload["study_campaign_pass"] = False
    pilot.write_text(json.dumps(payload))
    blocked = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert blocked.returncode != 0


def test_domain_pilot_uses_every_artifact_alpha_ar_pair(tmp_path):
    _, rows = make_manifest(tmp_path, "domain-pilot", _test_artifact(tmp_path))
    assert len(rows) == 4 * 7 * 4
    assert {float(row["alpha"]) for row in rows} == {0.5, 0.8, 0.95, 1.0}
    assert {float(row["aspect_ratio"]) for row in rows} == {
        1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0}


def test_stability_design_brackets_roots_on_the_measured_collision_horizon(tmp_path):
    _, rows = make_manifest(tmp_path, "stability", _test_artifact(tmp_path))
    assert len(rows) == 144
    assert {float(row["initial_theta"]) for row in rows} == {0.025, 0.225, 1.5}
    assert all(float(row["initial_theta"]) in (
        (0.025, 1.5) if float(row["aspect_ratio"]) <= 1.35 else (0.225, 1.5))
        for row in rows)
    # The long-time requirement rides the collision clock, not accumulated
    # cooling: the measured attractor relaxation is flat in alpha, so the run
    # length must not scale as 1/(1-alpha**2).
    assert {float(row["tau_end"]) for row in rows} == {1500.0}
    assert {float(row["sample_start_tau"]) for row in rows} == {500.0}
    assert {float(row["sample_delta_tau"]) for row in rows} == {5.0}
    assert {float(row["relaxation_horizon"]) for row in rows} == {390.0}
    for row in rows:
        assert float(row["sample_start_tau"]) >= float(row["relaxation_horizon"])
    # The screen must resolve what it tests, so it runs at production N.
    assert {int(row["particles"]) for row in rows} == {10000}
    assert {row["arm"] for row in rows} == {"scaled"}
    assert {float(row["dt"]) for row in rows if row["arm"] == "scaled"} == {0.0025}


def test_production_grid_excludes_the_out_of_domain_near_sphere_corner(tmp_path):
    artifact = _test_artifact(tmp_path)
    for mode in ("sweep", "stability"):
        _, rows = make_manifest(tmp_path / mode, mode, artifact)
        cases = {(float(row["alpha"]), float(row["aspect_ratio"]))
                 for row in rows}
        # alpha=0.5, AR=1.1 drives a20 past the artifact's a2_tr hull, so half
        # its evaluation-window closure queries fall back to the base law.
        assert (0.50, 1.10) not in cases
        assert (0.50, 1.20) in cases and (0.80, 1.10) in cases
        assert len(cases) == 36


def test_stability_sentinel_repeats_long_horizon_at_half_dt(tmp_path):
    _, rows = make_manifest(
        tmp_path, "stability-sentinel", _test_artifact(tmp_path))
    assert len(rows) == 80
    assert len({(row["alpha"], row["aspect_ratio"]) for row in rows}) == 5
    assert len({int(row["seed"]) for row in rows}) == 4
    assert {row["arm"] for row in rows} == {"scaled", "dt_half"}
    assert {float(row["initial_theta"]) for row in rows} == {0.025, 0.225, 1.5}
    assert {float(row["tau_end"]) for row in rows} == {1500.0}
    assert {int(row["particles"]) for row in rows} == {10000}
    assert {float(row["dt"]) for row in rows if row["arm"] == "dt_half"} == {0.00125}


def test_stability_requires_complete_passing_sentinel_on_same_bytes(tmp_path):
    artifact = _test_artifact(tmp_path)
    manifest, _ = make_manifest(tmp_path, "stability", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    pilot = tmp_path / "sentinel.json"
    command = [sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
               "--manifest", str(manifest), "--artifact", str(artifact),
               "--pilot-summary", str(pilot)]
    pilot.write_text(json.dumps({
        "mode": "stability-sentinel", "protocol_version": "hcs-ng-v8",
        "analysis_revision": "hcs-ng-analysis-v3",
        "orientation_integrator": "symmetric_midpoint_v1",
        "model_variant": "baseline", "invariant_corrections": False,
        "long_time_stability_campaign_pass": True,
        "engineering_control_coverage_pass": True,
        "artifact_sha256": digest,
        "n_tasks": 80, "n_completed_tasks": 80,
        "failed_tasks": [], "missing_tasks": [],
        "arm_dt": {"scaled": [0.0025], "dt_half": [0.00125]},
    }))
    accepted = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert accepted.returncode == 0, accepted.stderr
    payload = json.loads(pilot.read_text())
    payload["long_time_stability_campaign_pass"] = False
    pilot.write_text(json.dumps(payload))
    blocked = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert blocked.returncode != 0
    payload["long_time_stability_campaign_pass"] = True
    payload.pop("analysis_revision")
    pilot.write_text(json.dumps(payload))
    stale_analysis = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True)
    assert stale_analysis.returncode != 0


def test_sentinel_requires_passing_v7_engineering_gate(tmp_path):
    artifact = _test_artifact(tmp_path)
    manifest, _ = make_manifest(tmp_path, "stability-sentinel", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    summary = tmp_path / "engineering.json"
    cases = ((0.50, 1.35), (0.50, 3.0), (0.80, 2.0),
             (0.95, 2.0), (1.00, 3.0))
    summary.write_text(json.dumps({
        "mode": "engineering", "protocol_version": "hcs-ng-v8",
        "orientation_integrator": "symmetric_midpoint_v1",
        "model_variant": "baseline", "invariant_corrections": False,
        "study_campaign_pass": True, "artifact_sha256": digest,
        "n_tasks": 120, "n_completed_tasks": 120,
        "arm_dt": {"scaled": [0.0025], "unscaled": [0.0025],
                   "dt_half": [0.00125]},
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
    payload = json.loads(summary.read_text())
    payload["protocol_version"] = "hcs-ng-v6"
    summary.write_text(json.dumps(payload))
    stale = subprocess.run([
        sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
        "--manifest", str(manifest), "--artifact", str(artifact),
        "--pilot-summary", str(summary),
    ], cwd=ROOT, text=True, capture_output=True)
    assert stale.returncode != 0


def test_current_artifact_allows_numerics_but_gates_engineering(tmp_path):
    artifact = _test_artifact(tmp_path)
    numerics, _ = make_manifest(tmp_path, "numerics-pilot", artifact)
    allowed = subprocess.run([
        sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
        "--manifest", str(numerics), "--artifact", str(artifact),
        "--allow-engineering"], cwd=ROOT, text=True, capture_output=True)
    assert allowed.returncode == 0, allowed.stderr

    engineering, _ = make_manifest(tmp_path, "engineering", artifact)
    gated = subprocess.run([
        sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
        "--manifest", str(engineering), "--artifact", str(artifact)],
        cwd=ROOT, text=True, capture_output=True)
    assert gated.returncode != 0 and "pilot-summary" in gated.stderr

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
        tmp_path, "numerics-pilot", artifact, model_variant="angular_evidence")
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
    assert module.MAXIMUM_BULK_TO_THERMAL_TEMPERATURE_RATIO == 1.0e-12
    assert module.MAXIMUM_EVALUATION_CORRECTION_FALLBACK_FRACTION == 1.0e-2


def test_terminal_sampling_recovery_is_narrow_and_auditable():
    module = _analysis_module()
    series = {"tau": np.linspace(100.0, 190.0, 10)}
    summary = {
        "run_status": "complete", "sampling_eligible": True,
        "sampling_complete": False, "expected_samples": 11,
        "n_samples": 10, "sample_start_tau": 100.0,
        "sample_end_tau": 200.0, "sample_delta_tau": 10.0,
    }
    recovered = module.sampling_completion(summary, series)
    assert recovered["complete"]
    assert recovered["terminal_coverage_recovered"]
    two_missing = dict(summary, expected_samples=12)
    assert not module.sampling_completion(two_missing, series)["complete"]
    gapped = {"tau": np.array([100.0, 110.0, 120.0, 150.0, 160.0,
                                170.0, 180.0, 185.0, 188.0, 190.0])}
    assert not module.sampling_completion(summary, gapped)["complete"]


def test_sentinel_inherits_precision_but_keeps_hard_effect_cap():
    module = _analysis_module()
    noisy_consistent = np.array([-0.010, 0.000, 0.010, 0.018])
    engineering = module.paired_control_result(
        noisy_consistent, "a02", "engineering")
    sentinel = module.paired_control_result(
        noisy_consistent, "a02", "stability-sentinel")
    assert engineering["statistical_consistency_pass"]
    assert not engineering["precision_pass"]
    assert not engineering["pass"]
    assert sentinel["hard_effect_cap_pass"]
    assert not sentinel["precision_required_for_pass"]
    assert sentinel["pass"]
    too_few = module.paired_control_result(
        np.array([0.001]), "a02", "stability-sentinel")
    assert not too_few["replicate_count_pass"]
    assert not too_few["pass"]
    too_large = module.paired_control_result(
        np.array([-0.020, 0.010, 0.030, 0.040]),
        "a02", "stability-sentinel")
    assert not too_large["hard_effect_cap_pass"]
    assert not too_large["pass"]


def test_stationarity_gate_detects_delayed_departure():
    module = _analysis_module()
    too_short = module.replicate_stationarity(
        [np.zeros(module.MINIMUM_STATIONARITY_SAMPLES - 1)] * 2,
        module.LOG_THETA)
    assert not too_short["pass"]
    assert too_short["reason"] == "too_few_late_samples"
    flat = np.zeros(40)
    delayed = flat.copy()
    delayed[30:] = np.linspace(0.0, 0.12, 10)
    assert module.replicate_stationarity([flat, flat], module.LOG_THETA)["pass"]
    verdict = module.replicate_stationarity(
        [delayed, delayed], module.LOG_THETA)
    assert not verdict["pass"]
    assert verdict["reason"] == "late_window_drift_resolved"
    unresolved_up = flat.copy(); unresolved_up[27:] = 0.08
    unresolved_down = flat.copy(); unresolved_down[27:] = -0.08
    unresolved = module.replicate_stationarity(
        [unresolved_up, unresolved_down], module.LOG_THETA)
    assert not unresolved["pass"]
    assert unresolved["reason"] == "late_window_drift_under_resolved"


def test_a_run_stopped_by_the_time_ceiling_is_not_a_complete_case(tmp_path):
    """Protocol v7 pinned t_end=1e5 while asking for tau_end=6154 at
    alpha=0.95.  Near-sphere rods collide several times less often per unit
    time, so four tasks exited on the physical-time ceiling at cpp=4872 after
    a full 16-hour allocation and still reported ``run_status: complete``.
    The truncation must now name itself and fail the case closed."""
    module = _analysis_module()
    prefix = tmp_path / "task"
    (tmp_path / "task_ng_moments.csv").write_text("tau,a20\n1.0,0.0\n")
    row = {"task_id": "0", "output_prefix": str(prefix), "replicate": "0",
           "seed": "1", "initial_theta": "1.0", "tau_end": "6153.8",
           "orientation_integrator": "symmetric_midpoint_v1"}
    payload = {"run_status": "complete", "cpp": 4872.4,
               "orientation_integrator": "symmetric_midpoint_v1",
               "non_gaussian": {}}
    for reason, accepted in (("collision_target", True),
                             ("time_ceiling", False),
                             (None, True)):     # legacy result: no field
        body = dict(payload)
        if reason is not None:
            body["termination_reason"] = reason
        Path(str(prefix) + ".json").write_text(json.dumps(body))
        try:
            module.replicate_record(row)
            raised = False
        except RuntimeError as error:
            raised = True
            assert "time_ceiling" in str(error)
        assert raised != accepted, reason


def test_drift_resolution_is_not_taken_from_two_replicates():
    """Analysis v2 built the tolerance from the spread of two drift values --
    a standard error with one degree of freedom -- and then multiplied it by
    3.0 as if it were a normal quantile.  Two realizations that happen to
    agree collapsed the tolerance and failed a quiet run; two that happen to
    disagree inflated it past the ceiling.  v3 must take its resolution from
    within-realization precision instead."""
    module = _analysis_module()
    rng = np.random.default_rng(20260925)
    noise = 0.004

    # Identical-by-luck replicates must not manufacture a near-zero tolerance.
    twin = rng.normal(0.0, noise, 60)
    lucky = module.replicate_stationarity([twin, twin.copy()], "a20")
    assert lucky["pass"]
    assert lucky["statistical_resolution_3se"] > 1.0e-6
    assert lucky["resolution_source"] == "detrended_autocorrelation_consistent"

    # A constant per-seed offset is not a drift and must not be charged as one.
    offset = module.replicate_stationarity(
        [rng.normal(0.0, noise, 60) + 0.05,
         rng.normal(0.0, noise, 60) - 0.05], "a20")
    assert offset["pass"]

    # Real power is kept: a genuine trend across the window is still caught.
    ramp = np.linspace(0.0, 0.05, 60)
    drifting = module.replicate_stationarity(
        [ramp + rng.normal(0.0, noise, 60),
         ramp + rng.normal(0.0, noise, 60)], "a20")
    assert not drifting["pass"]
    assert drifting["reason"] == "late_window_drift_resolved"


def test_stationarity_false_rejection_rate_is_negligible_at_production_scale():
    """The v7 screen ran N=2000 with 40 late samples, where per-snapshot
    scatter (sd(a20) = 0.016-0.065) was several times the 0.01 drift it was
    asked to resolve.  At the v8 window and particle count a stationary
    coordinate must essentially never be rejected."""
    module = _analysis_module()
    rng = np.random.default_rng(902)
    rejected = 0
    trials = 120
    for _ in range(trials):
        curves = [rng.normal(0.0, 0.013, 201) for _ in range(2)]
        if not module.replicate_stationarity(curves, "a20")["pass"]:
            rejected += 1
    assert rejected <= 2, f"{rejected}/{trials} stationary coordinates rejected"


def test_stage_verifier_names_the_failing_coordinates(tmp_path):
    """A bare AssertionError after a 14-hour allocation costs a diagnosis
    round-trip.  The verifier must say which coordinate failed which gate."""
    verify = ROOT / "hpc/verify_hcs_ng_stage.py"
    summary = tmp_path / "summary.json"
    passing = {
        "protocol_version": "hcs-ng-v8",
        "analysis_revision": "hcs-ng-analysis-v3",
        "mode": "stability", "n_tasks": 144, "n_completed_tasks": 144,
        "failed_tasks": [], "missing_tasks": [],
        "artifact_sha256": "d" * 64,
        "long_time_stability_campaign_pass": True,
        "two_sided_attraction_pass": True,
        "cases": [{"alpha": 0.5, "aspect_ratio": 1.2, "initial_theta": 0.025,
                   "sampling_complete": True, "runtime_pass": True,
                   "stationarity_pass": True, "dissipation_horizon_pass": True,
                   "correction_fallback_pass": True}],
    }
    command = [sys.executable, str(verify), "stability", str(summary),
               "--artifact-sha256", "d" * 64]
    summary.write_text(json.dumps(passing))
    assert subprocess.run(command, cwd=ROOT).returncode == 0

    broken = json.loads(json.dumps(passing))
    broken["cases"][0]["stationarity_pass"] = False
    broken["long_time_stability_campaign_pass"] = False
    summary.write_text(json.dumps(broken))
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "alpha=0.50 AR=1.20" in result.stderr
    assert "stationarity_pass" in result.stderr

    # A stale summary from the previous protocol must not authorize anything.
    stale = json.loads(json.dumps(passing))
    stale["protocol_version"] = "hcs-ng-v7"
    summary.write_text(json.dumps(stale))
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0

    # Neither may a summary built from different artifact bytes.
    other = json.loads(json.dumps(passing))
    other["artifact_sha256"] = "e" * 64
    summary.write_text(json.dumps(other))
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0


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


def test_sweep_design_is_two_sided_and_includes_elastic_control(tmp_path):
    _, rows = make_manifest(tmp_path, "sweep", _test_artifact(tmp_path))
    cases = {(float(row["alpha"]), float(row["aspect_ratio"])) for row in rows}
    assert len(cases) == 36 and len(rows) == 36 * 10
    assert min(ar for _, ar in cases) == 1.1
    assert {float(row["dt"]) for row in rows if row["arm"] == "scaled"} == {0.0025}
    assert {alpha for alpha, ar in cases if ar == 2.0} == {
        0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0}
    assert {ar for alpha, ar in cases if alpha == 0.8} == {
        1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0}
    assert {int(row["particles"]) for row in rows} == {10000}
    assert {row["arm"] for row in rows} == {"scaled"}
    # Production carries the attraction test: five realizations from each of
    # the two bracketing starts, not ten from one.
    assert {float(row["initial_theta"]) for row in rows} == {0.025, 0.225, 1.5}
    for case in cases:
        starts = {float(row["initial_theta"]) for row in rows
                  if (float(row["alpha"]), float(row["aspect_ratio"])) == case}
        assert len(starts) == 2 and max(starts) == 1.5
        assert len([row for row in rows
                    if (float(row["alpha"]),
                        float(row["aspect_ratio"])) == case]) == 10


def test_ng_analysis_runs_after_any_array_outcome():
    submit = (ROOT / "hpc/submit_hcs_ng_campaign.sh").read_text()
    assert '--dependency="afterany:$JOB"' in submit
    assert "--allow-missing" in (ROOT / "hpc/analyze_hcs_ng.slurm").read_text()


def test_sweep_requires_passing_long_time_stability_on_same_bytes(tmp_path):
    # The two-sided production sweep starts every AR <= 1.35 coordinate from
    # theta0 = 0.025, so it now depends on the sampler's low-theta extension
    # in a way the old one-sided (theta0 = 1) design did not.
    surface = np.array([[alpha, theta, ar]
                        for alpha in (0.5, 0.8, 0.95, 1.0)
                        for theta in (0.025, 0.2, 1.0, 2.0)
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
            ({"mode": "engineering", "protocol_version": "hcs-ng-v8",
              "orientation_integrator": "symmetric_midpoint_v1",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": True,
              "artifact_sha256": digest}, False),
            ({"mode": "stability", "protocol_version": "hcs-ng-v6",
              "orientation_integrator": "symmetric_midpoint_v1",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": True,
              "artifact_sha256": digest}, False),
            ({"mode": "stability", "protocol_version": "hcs-ng-v8",
              "orientation_integrator": "symmetric_midpoint_v1",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": False,
              "analysis_revision": "hcs-ng-analysis-v3",
              "n_tasks": 144, "n_completed_tasks": 144,
              "two_sided_attraction_pass": True,
              "failed_tasks": [], "missing_tasks": [],
              "artifact_sha256": digest}, False),
            ({"mode": "stability", "protocol_version": "hcs-ng-v8",
              "orientation_integrator": "symmetric_midpoint_v1",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": True,
              "artifact_sha256": "0" * 64}, False),
            ({"mode": "stability", "protocol_version": "hcs-ng-v8",
              "orientation_integrator": "symmetric_midpoint_v1",
              "model_variant": "baseline", "invariant_corrections": False,
              "long_time_stability_campaign_pass": True,
              "artifact_sha256": digest,
              "analysis_revision": "hcs-ng-analysis-v3",
              "n_tasks": 144, "n_completed_tasks": 144,
              "two_sided_attraction_pass": True,
              "failed_tasks": [], "missing_tasks": []}, True)):
        pilot.write_text(json.dumps(payload))
        result = subprocess.run(command + ["--pilot-summary", str(pilot)],
                                cwd=ROOT, text=True, capture_output=True)
        assert (result.returncode == 0) == ok, result.stderr


def test_two_sided_sweep_requires_the_low_theta_sampler_extension(tmp_path):
    """The old theta0=1 sweep never queried the low-theta hull; the two-sided
    design does, so an artifact without that extension must fail closed."""
    narrow = np.array([[alpha, theta, ar]
                       for alpha in (0.5, 0.8, 0.95, 1.0)
                       for theta in (0.2, 1.0, 2.0)
                       for ar in (1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0)])
    artifact = tmp_path / "narrow.npz"
    np.savez_compressed(artifact, surface_coordinates=narrow)
    manifest, _ = make_manifest(tmp_path, "sweep", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({
        "mode": "stability", "protocol_version": "hcs-ng-v8",
        "analysis_revision": "hcs-ng-analysis-v3",
        "orientation_integrator": "symmetric_midpoint_v1",
        "model_variant": "baseline", "invariant_corrections": False,
        "long_time_stability_campaign_pass": True,
        "two_sided_attraction_pass": True, "artifact_sha256": digest,
        "n_tasks": 144, "n_completed_tasks": 144,
        "failed_tasks": [], "missing_tasks": []}))
    blocked = subprocess.run(
        [sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
         "--manifest", str(manifest), "--artifact", str(artifact),
         "--pilot-summary", str(pilot)], cwd=ROOT, text=True, capture_output=True)
    assert blocked.returncode != 0
    assert "hull misses campaign cases" in blocked.stderr


def test_tails_requires_complete_passing_sweep_on_same_bytes(tmp_path):
    artifact = _test_artifact(tmp_path)
    manifest, _ = make_manifest(tmp_path, "tails", artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    summary = tmp_path / "sweep.json"
    command = [sys.executable, str(ROOT / "hpc/check_hcs_ng_prerequisites.py"),
               "--manifest", str(manifest), "--artifact", str(artifact),
               "--pilot-summary", str(summary)]
    payload = {
        "mode": "sweep", "protocol_version": "hcs-ng-v8",
        "orientation_integrator": "symmetric_midpoint_v1",
        "model_variant": "baseline", "invariant_corrections": False,
        "study_campaign_pass": True, "scientific_outputs_released": True,
        "artifact_sha256": digest,
        "n_tasks": 360, "n_completed_tasks": 360,
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
