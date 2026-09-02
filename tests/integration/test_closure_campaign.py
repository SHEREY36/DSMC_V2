import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "make_closure_manifest", Path(__file__).resolve().parents[2] / "hpc" / "make_closure_manifest.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ESTIMATE_SPEC = importlib.util.spec_from_file_location(
    "estimate_closure_task",
    Path(__file__).resolve().parents[2] / "hpc" / "estimate_closure_task.py")
ESTIMATE_MODULE = importlib.util.module_from_spec(ESTIMATE_SPEC)
ESTIMATE_SPEC.loader.exec_module(ESTIMATE_MODULE)


class ClosureCampaignTests(unittest.TestCase):
    def test_sentinel_slurm_job_uses_submit_directory_not_spool_path(self):
        root = Path(__file__).resolve().parents[2]
        script = (root / "job_closure_sentinel.slurm").read_text()
        self.assertIn("#SBATCH --array=0-35%12", script)
        self.assertIn("ROOT=${SLURM_SUBMIT_DIR:-$PWD}", script)
        self.assertNotIn('dirname "$0"', script)

    def test_sentinel_has_36_baseline_nodes(self):
        rows = MODULE.grid_rows("sentinel", 5000, 0, "results/ctc_closure")
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["ensemble_id"] for row in rows}, {0})

    def test_full_baseline_grid_matches_requested_axes(self):
        rows = MODULE.grid_rows("baseline", 5000, 0, "results/ctc_closure")
        self.assertEqual(len(rows), 13 * 16 * 6)
        self.assertIn(1.0, {row["alpha"] for row in rows})
        self.assertIn(2.0, {row["theta"] for row in rows})

    def test_excitation_design_has_47_ensembles_and_elastic_sentinel(self):
        rows = MODULE.grid_rows("excitation", 5000, 0, "results/ctc_closure")
        self.assertEqual(len(MODULE.ENSEMBLES), 47)
        self.assertEqual(len(rows), 3 * 4 * 3 * 47 + 1)
        elastic = [row for row in rows if row["role"] == "elastic_sentinel"]
        self.assertEqual(len(elastic), 1)
        self.assertEqual((elastic[0]["alpha"], elastic[0]["theta"],
                          elastic[0]["aspect_ratio"]), (1.0, 1.0, 2.0))

    def test_samples_are_capped_at_200000(self):
        rows = MODULE.grid_rows("sentinel", 200000, 0, "results/ctc_closure")
        self.assertTrue(all(row["nsamples"] == 200000 for row in rows))

    def test_scientific_fit_error_is_serializable_gate_failure(self):
        row = MODULE.grid_rows("sentinel", 5000, 0, "results/ctc_closure")[0]
        result = ESTIMATE_MODULE._failed_fit_result(
            row, Path(row["output_directory"]), ValueError("infeasible moments on (0,1)"))
        self.assertFalse(result["qa"]["sentinel_pass"])
        self.assertFalse(result["qa"]["precision_pass"])
        self.assertEqual(result["fit_error"]["type"], "ValueError")
        self.assertIn("scientific_fit_error:ValueError",
                      result["qa"]["continuation_reasons"])

    def test_summary_fails_closed_when_an_expected_estimate_is_missing(self):
        root = Path(__file__).resolve().parents[2]
        script = root / "hpc" / "summarize_sentinel.py"
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            run0, run1 = temporary / "run0", temporary / "run1"
            run0.mkdir(); run1.mkdir()
            runtime = {"hits_per_second": 1.0, "attempts_per_hit": 2.0}
            (run0 / "runtime_v2.json").write_text(json.dumps(runtime))
            (run1 / "runtime_v2.json").write_text(json.dumps(runtime))
            rows = MODULE.grid_rows("sentinel", 5000, 0, str(temporary))[:2]
            rows[0]["output_directory"] = str(run0)
            rows[1]["output_directory"] = str(run1)
            manifest = temporary / "manifest.csv"
            import csv
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MODULE.FIELDS)
                writer.writeheader(); writer.writerows(rows)
            estimates = temporary / "estimates"
            estimates.mkdir()
            row = rows[0]
            target = estimates / (
                f"alpha_{float(row['alpha']):.3f}_theta_{float(row['theta']):.3f}_"
                f"AR_{float(row['aspect_ratio']):.3f}_ensemble_000.json")
            target.write_text(json.dumps({"qa": {"sentinel_pass": True}}))
            report = temporary / "report.json"
            subprocess.run([
                sys.executable, str(script), "--runs-root", str(temporary),
                "--estimates", str(estimates), "--manifest", str(manifest),
                "--output", str(report),
            ], check=True, capture_output=True, text=True)
            payload = json.loads(report.read_text())
            self.assertFalse(payload["analysis_complete"])
            self.assertFalse(payload["all_sentinel_gates_pass"])
            self.assertIn("missing_node_estimate", payload["failures"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
