import os
import sys
import subprocess
import tempfile
import unittest
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpc"))

from estimate_task import node_paths  # noqa: E402
from make_manifest import base_rows  # noqa: E402


class HPCStageTests(unittest.TestCase):
    def test_pilot_combined_and_continuation_paths_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results" / "ctc"
            pilot = Path(base_rows("pilot", 20_000, 0, str(root))[0]["output_directory"])
            production = Path(base_rows("production", 80_000, 1, str(root))[0]["output_directory"])
            continuation = root / "continuation" / (
                "alpha_0.500_theta_0.100_AR_1.100_shard_02")
            for path in (pilot, production, continuation):
                path.mkdir(parents=True)
                (path / "_SUCCESS").touch()
            _, pilot_paths = node_paths(0, "pilot", root)
            _, combined_paths = node_paths(0, "combined", root)
            self.assertEqual(pilot_paths, [pilot])
            self.assertEqual(combined_paths, [pilot, production, continuation])

    def test_estimator_reads_legacy_cr_suffixed_pilot_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results" / "ctc"
            expected = Path(base_rows("pilot", 20_000, 0, str(root))[0]["output_directory"])
            legacy = Path(str(expected) + "\r")
            legacy.mkdir(parents=True)
            (legacy / "_SUCCESS").touch()
            _, paths = node_paths(0, "pilot", root)
            self.assertEqual(paths, [legacy])

    def test_manifest_has_unix_line_endings(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "pilot.csv"
            subprocess.run([
                sys.executable, str(ROOT / "hpc" / "make_manifest.py"),
                "--stage", "pilot", "--output", str(manifest),
            ], check=True, capture_output=True, text=True)
            self.assertNotIn(b"\r", manifest.read_bytes())

    def test_all_root_jobs_target_the_negishi_account(self):
        jobs = sorted(ROOT.glob("job_*.slurm"))
        self.assertGreaterEqual(len(jobs), 8)
        for job in jobs:
            text = job.read_text()
            self.assertIn("#SBATCH -A morri353", text, job.name)
            self.assertIn("#SBATCH -p cpu", text, job.name)

    def test_artifact_job_uses_submission_directory_not_slurm_spool(self):
        script = (ROOT / "hpc" / "aggregate.slurm").read_text()
        self.assertIn("SLURM_SUBMIT_DIR", script)

    def test_artifact_precompute_fills_256_core_account_without_oversubscription(self):
        worker = (ROOT / "hpc" / "artifact_precompute_stride.slurm").read_text()
        submitter = (ROOT / "hpc" / "submit_postfit_artifact.sh").read_text()
        self.assertIn("#SBATCH --cpus-per-task=2", worker)
        self.assertIn("OPENBLAS_NUM_THREADS=1", worker)
        self.assertIn("ARTIFACT_MAX_CORES:-256", submitter)
        self.assertIn("afterok:$PRE_JOB", submitter)
        self.assertIn("artifact_precompute_stride.slurm", submitter)
        self.assertNotIn("{print $3; exit}", submitter)
        self.assertIn("|| MAX_ARRAY=1000", submitter)

    def test_artifact_aggregation_consumes_parallel_precompute_payloads(self):
        script = (ROOT / "hpc" / "aggregate.slurm").read_text()
        self.assertIn("--precomputed-directory", script)
        self.assertIn("OPENBLAS_NUM_THREADS=1", script)

    def test_hcs_gate_has_two_initial_conditions_and_known_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "hcs.csv"
            subprocess.run([
                sys.executable,
                str(ROOT / "DSMC_0D_v2" / "scripts" / "make_hcs_validation_manifest.py"),
                "--output", str(manifest), "--results", str(Path(temporary) / "results"),
            ], check=True, capture_output=True, text=True)
            with manifest.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 36)
            ar2_alpha95 = [row for row in rows
                           if float(row["alpha"]) == 0.95
                           and float(row["aspect_ratio"]) == 2.0]
            self.assertEqual({float(row["theta0"]) for row in ar2_alpha95}, {0.75, 1.25})
            self.assertEqual({float(row["target_theta"]) for row in ar2_alpha95}, {0.9792})
            self.assertEqual({int(row["replicate"]) for row in ar2_alpha95}, {0, 1, 2})

    def test_negishi_environment_includes_hcs_plot_dependency(self):
        setup = (ROOT / "hpc" / "setup_negishi_env.sh").read_text()
        submitter = (ROOT / "hpc" / "submit_hcs_plot.sh").read_text()
        self.assertIn("'matplotlib>=3.7'", setup)
        self.assertIn("import matplotlib", setup)
        self.assertIn("import matplotlib", submitter)
        self.assertIn("does not rerun DSMC", submitter)

    def test_hcs_drift_gate_uses_the_replicate_mean(self):
        plot_path = (ROOT / "DSMC_0D_v2" / "scripts"
                     / "plot_hcs_validation.py")
        spec = importlib.util.spec_from_file_location("plot_hcs_validation", plot_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        tau = np.linspace(0.0, 20.0, 101)
        flat = np.column_stack((tau, np.ones_like(tau), np.ones_like(tau)))
        noisy_theta = np.ones_like(tau)
        noisy_theta[70:] = np.linspace(1.0, 1.12, len(tau) - 70)
        noisy = np.column_stack((tau, noisy_theta, np.ones_like(tau)))

        start = int(0.7 * len(tau))
        individual_drift = module.relative_linear_drift(
            tau[start:], noisy_theta[start:])
        mean_drift = module.replicate_mean_relative_drift([flat, flat, noisy])
        self.assertGreater(individual_drift, 0.10)
        self.assertLess(mean_drift, 0.10)

    def test_excitation_pilots_are_bounded_and_parallel(self):
        with tempfile.TemporaryDirectory() as temporary:
            for mode, expected in (("hcs-pilot", 96), ("full-pilot", 72),
                                   ("correction-grid", 1296),
                                   ("usf-extension", 1296),
                                   ("production-grid", 10368),
                                   ("independent-holdout", 36)):
                manifest = Path(temporary) / f"{mode}.csv"
                subprocess.run([
                    sys.executable, str(ROOT / "hpc" / "make_excitation_manifest.py"),
                    "--mode", mode, "--grid", str(ROOT / "manifests" / "artifact_grid.csv"),
                    "--estimates", str(ROOT / "results" / "closure_estimates" / "artifact_grid"),
                    "--output", str(manifest), "--results", str(Path(temporary) / mode),
                ], check=True, capture_output=True, text=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "contracts" / "python")
                        + ":" + str(ROOT / "Coll_Models_v2" / "src")})
                with manifest.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), expected)
                self.assertGreaterEqual(min(float(row["eta"]) for row in rows), -0.5)
                self.assertLessEqual(max(float(row["eta"]) for row in rows), 0.5)
                if mode == "usf-extension":
                    coordinates = {
                        (float(row["alpha"]), float(row["theta"]),
                         float(row["aspect_ratio"])) for row in rows}
                    self.assertEqual(len(coordinates), 18)
                    self.assertEqual(min(item[0] for item in coordinates), 0.5)
                    self.assertEqual(min(item[2] for item in coordinates), 1.5)
        worker = (ROOT / "hpc" / "excitation_fit_stride.slurm").read_text()
        submitter = (ROOT / "hpc" / "submit_excitation_campaign.sh").read_text()
        self.assertIn("#SBATCH --cpus-per-task=1", worker)
        self.assertIn("EXCITATION_MAX_CORES:-256", submitter)
        self.assertIn("require_hcs_pass.py", submitter)
        self.assertIn("require_excitation_pass.py", submitter)

    def test_independent_holdout_uses_fresh_ctc_and_boundary_points_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            ctc = Path(temporary) / "ctc.csv"
            fit = Path(temporary) / "fit.csv"
            subprocess.run([
                sys.executable,
                str(ROOT / "hpc" / "make_independent_ctc_holdout_manifest.py"),
                "--ctc-output", str(ctc), "--fit-output", str(fit),
                "--results-root", str(Path(temporary) / "raw"),
            ], check=True, capture_output=True, text=True)
            with ctc.open(newline="") as handle:
                ctc_rows = list(csv.DictReader(handle))
            with fit.open(newline="") as handle:
                fit_rows = list(csv.DictReader(handle))
            self.assertEqual(len(ctc_rows), 2)
            self.assertEqual(len(fit_rows), 1)
            self.assertEqual(ctc_rows[0]["seed"], ctc_rows[1]["seed"])
            self.assertEqual(fit_rows[0]["anchor_directory"],
                             ctc_rows[1]["output_directory"])
        submitter = (ROOT / "hpc" / "submit_independent_ctc_holdout.sh").read_text()
        self.assertIn("--array=0-35%36", submitter)
        self.assertIn("EXCITATION_BOOTSTRAP=0", submitter)
        self.assertIn("CLOSURE_PROPENSITY_WORKERS=20", submitter)

    def test_usf_candidate_pipeline_caches_then_fits_and_gates(self):
        submitter = (ROOT / "hpc" / "submit_usf_candidate_pipeline.sh").read_text()
        propensity = (ROOT / "hpc" / "excitation_propensity_array.slurm").read_text()
        self.assertIn("--cpus-per-task=12", propensity)
        self.assertIn("EXCITATION_OFFSETS:-128", propensity)
        self.assertIn('dependency="afterok:$PROP_JOB"', submitter)
        self.assertIn('dependency="afterok:$EXT_QA_JOB"', submitter)
        self.assertIn('dependency="afterok:$HCS_PLOT_JOB:$HOLDOUT_JOB"', submitter)
        self.assertIn("check_usf_candidate.slurm", submitter)
        self.assertIn("paired_usf_pilot_job", submitter)

    def test_usf_manifests_are_paired_pilot_and_corrected_full_grid(self):
        script = ROOT / "DSMC_0D_v2" / "scripts" / "make_usf_validation_manifest.py"
        with tempfile.TemporaryDirectory() as temporary:
            for mode, expected, arms in (
                    ("pilot", 24, {"corrected", "uncorrected"}),
                    ("full", 160, {"corrected"})):
                manifest = Path(temporary) / f"{mode}.csv"
                subprocess.run([
                    sys.executable, str(script), "--mode", mode,
                    "--output", str(manifest),
                    "--results", str(Path(temporary) / mode),
                ], check=True, capture_output=True, text=True)
                with manifest.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), expected)
                self.assertEqual({row["arm"] for row in rows}, arms)
                if mode == "pilot":
                    pairs = {(row["alpha"], row["aspect_ratio"], row["replicate"])
                             for row in rows}
                    self.assertTrue(all(sum(
                        (candidate["alpha"], candidate["aspect_ratio"],
                         candidate["replicate"]) == pair for candidate in rows) == 2
                        for pair in pairs))

    def test_usf_preflight_is_bound_to_exact_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "candidate.npz"
            artifact.write_bytes(b"candidate-a")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            hcs = root / "hcs.json"
            holdout = root / "holdout.json"
            hcs.write_text(json.dumps({
                "physics_gate_pass": True, "artifact_sha256": digest}))
            holdout.write_text(json.dumps({
                "validation_pass": True, "artifact_sha256": digest}))
            command = [
                sys.executable, str(ROOT / "hpc" / "require_usf_prerequisites.py"),
                "--hcs", str(hcs), "--holdout", str(holdout),
                "--artifact", str(artifact),
            ]
            self.assertEqual(subprocess.run(command).returncode, 0)
            artifact.write_bytes(b"candidate-b")
            self.assertNotEqual(subprocess.run(command).returncode, 0)


if __name__ == "__main__":
    unittest.main()
