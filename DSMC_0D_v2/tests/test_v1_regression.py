import tempfile
import unittest
from pathlib import Path

from dsmc_v2.simulation import run_simulation


class V1RegressionTests(unittest.TestCase):
    def test_all_legacy_seeded_hcs_matches_v1_golden_output(self):
        model_root = Path(__file__).resolve().parents[1] / "models"
        config = {
            "particle": {"AR": 2.0, "radius": 0.5, "mass": 1.0},
            "system": {"kTt": 1.0, "kTr": 1.0, "alpha": 0.8, "eta": 1.0,
                       "phi": 0.01, "domain": [20.0, 20.0, 20.0], "C_alpha": 1.0},
            "time": {"dt": 0.01, "dtau": 0.1, "t_end": 2.0, "tau_end": None,
                     "equilibration_time": 0.0},
            "flow": {"mode": "hcs", "shear_rate": 0.0},
            "simulation": {"sphere_collision": False, "use_isotropic_eps": True},
            "preprocessing": {"model_root": str(model_root),
                              "dissipation": {"beta_a": 1.21, "beta_b": 3.67}},
            "microscopic_closure": {"routing": "legacy_rank0", "angular": "legacy"},
        }
        expected = (
            "     0.000000      0.000000      0.880180      0.977645      0.919166\n"
            "     1.070000      0.129032      0.841291      1.020970      0.913162\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hcs.txt"
            diagnostics = run_simulation(config, 42, output)
            self.assertEqual(output.read_text(), expected)
            self.assertEqual(diagnostics["collisions"], 12)
            self.assertAlmostEqual(diagnostics["cpp"], 12.0 / 62.0)


if __name__ == "__main__":
    unittest.main()
