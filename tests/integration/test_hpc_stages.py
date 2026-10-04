import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class HPCStageTests(unittest.TestCase):
    def test_artifact_job_uses_submission_directory_not_slurm_spool(self):
        script = (ROOT / "hpc" / "aggregate.slurm").read_text()
        self.assertIn("SLURM_SUBMIT_DIR", script)

    def test_artifact_precompute_worker_is_single_threaded(self):
        worker = (ROOT / "hpc" / "artifact_precompute_stride.slurm").read_text()
        submitter = (ROOT / "hpc" / "submit_final_campaign.sh").read_text()
        self.assertIn("#SBATCH --cpus-per-task=2", worker)
        self.assertIn("OPENBLAS_NUM_THREADS=1", worker)
        self.assertIn("artifact_precompute_stride.slurm", submitter)

    def test_artifact_aggregation_consumes_parallel_precompute_payloads(self):
        script = (ROOT / "hpc" / "aggregate.slurm").read_text()
        self.assertIn("--precomputed-directory", script)
        self.assertIn("OPENBLAS_NUM_THREADS=1", script)

    def test_negishi_environment_includes_plot_dependency(self):
        setup = (ROOT / "hpc" / "setup_negishi_env.sh").read_text()
        self.assertIn("'matplotlib>=3.7'", setup)
        self.assertIn("verify_python_environment.py", setup)

    def test_explicit_python_selection_never_silently_falls_back(self):
        launcher = (ROOT / "hpc" / "python.sh").read_text()
        self.assertIn("DSMC_V2_PYTHON is not executable", launcher)
        self.assertIn('exec "$DSMC_V2_PYTHON" "$@"', launcher)


if __name__ == "__main__":
    unittest.main()
