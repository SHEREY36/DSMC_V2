import os
import sys
import subprocess
import tempfile
import unittest
import csv
import importlib.util
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
                                   ("production-grid", 10368)):
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
        worker = (ROOT / "hpc" / "excitation_fit_stride.slurm").read_text()
        submitter = (ROOT / "hpc" / "submit_excitation_campaign.sh").read_text()
        self.assertIn("#SBATCH --cpus-per-task=1", worker)
        self.assertIn("EXCITATION_MAX_CORES:-256", submitter)
        self.assertIn("require_hcs_pass.py", submitter)
        self.assertIn("require_excitation_pass.py", submitter)


if __name__ == "__main__":
    unittest.main()
