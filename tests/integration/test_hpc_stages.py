import sys
import subprocess
import tempfile
import unittest
import csv
from pathlib import Path


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
            self.assertEqual(len(rows), 12)
            ar2_alpha95 = [row for row in rows
                           if float(row["alpha"]) == 0.95
                           and float(row["aspect_ratio"]) == 2.0]
            self.assertEqual({float(row["theta0"]) for row in ar2_alpha95}, {0.75, 1.25})
            self.assertEqual({float(row["target_theta"]) for row in ar2_alpha95}, {0.9792})


if __name__ == "__main__":
    unittest.main()
