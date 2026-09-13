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
    surface = np.array([[alpha, 1.0, ar]
                        for alpha in (0.5, 0.8, 0.95, 1.0)
                        for ar in (1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0)])
    beta = np.array([[alpha, theta, ar]
                     for alpha in (0.8, 0.95, 1.0)
                     for theta in (0.2, 1.0, 2.0)
                     for ar in (2.0, 3.0)])
    artifact = tmp_path / "closure.npz"
    np.savez_compressed(artifact, surface_coordinates=surface,
                        beta_coordinates=beta)
    return artifact


def make_manifest(tmp_path, mode, artifact):
    manifest = tmp_path / f"{mode}.csv"
    subprocess.run([
        sys.executable, str(MAKE), "--mode", mode, "--artifact", str(artifact),
        "--output", str(manifest), "--results", str(tmp_path / "results")],
        check=True, cwd=ROOT)
    with manifest.open(newline="") as handle:
        return manifest, list(csv.DictReader(handle))


def test_engineering_design_pairs_scaled_and_unscaled(tmp_path):
    _, rows = make_manifest(tmp_path, "engineering", _test_artifact(tmp_path))
    assert len(rows) == 24
    assert {row["arm"] for row in rows} == {"scaled", "unscaled"}
    assert len({(row["alpha"], row["aspect_ratio"]) for row in rows}) == 3
    assert {int(row["particles"]) for row in rows} == {2000}


def test_domain_pilot_uses_every_artifact_alpha_ar_pair(tmp_path):
    _, rows = make_manifest(tmp_path, "domain-pilot", _test_artifact(tmp_path))
    assert len(rows) == 4 * 7 * 4
    assert {float(row["alpha"]) for row in rows} == {0.5, 0.8, 0.95, 1.0}
    assert {float(row["aspect_ratio"]) for row in rows} == {
        1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0}


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
