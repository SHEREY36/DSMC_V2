import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coll_models_v2.projections import angular_quantiles, energy_quantiles
from dsmc_v2.artifact import VariationalClosure
from dsmc_v2.legacy_models import FrozenLossModel
from dsmc_v2.simulation import run_simulation, runtime_gate_status
from dsmc_v2_contracts import FEATURE_NAMES


class VariationalArtifactTests(unittest.TestCase):
    @staticmethod
    def _write(path, joint=False):
        coordinates = np.array([[a, t, r] for a in (0.8, 1.0)
                                 for t in (0.1, 3.0) for r in (1.5, 2.0)])
        probability = np.linspace(0.0, 1.0, 513)
        ep = np.zeros((len(coordinates), 2))
        ap = np.zeros((len(coordinates), 2))
        beta = np.zeros((len(coordinates), len(FEATURE_NAMES)))
        beta[:, 0] = 0.2
        np.savez_compressed(
            path, schema_version=np.array("2.2.0"),
            artifact_type=np.array("bl_variational_closure"),
            feature_names=np.array(FEATURE_NAMES), surface_coordinates=coordinates,
            p_exch=np.full(len(coordinates), 0.4), energy_parameters=ep,
            angular_parameters=ap, quantile_probability=probability,
            energy_quantiles=np.array([energy_quantiles(row, probability) for row in ep]),
            angular_quantiles=np.array([angular_quantiles(row, probability) for row in ap]),
            beta_coordinates=coordinates, beta=beta, beta_se=np.zeros_like(beta),
            beta_deployed=beta != 0.0, feature_lower=np.full(len(FEATURE_NAMES), -0.5),
            feature_upper=np.full(len(FEATURE_NAMES), 0.5),
            joint_deployed=np.full(len(coordinates), joint, dtype=bool),
            joint_parameters=(np.tile([0.1, -0.1, 0.3], (len(coordinates), 1))
                              if joint else np.full((len(coordinates), 3), np.nan)),
        )

    def test_load_interpolate_and_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            closure = VariationalClosure(path)
            features = np.zeros(len(FEATURE_NAMES)); features[0] = 0.1
            state = closure.kernel_state(0.9, 0.75, 1.75, features)
            self.assertAlmostEqual(state["p_exch"], 0.4)
            self.assertAlmostEqual(state["energy_parameters"][0], 0.02)
            rng = np.random.default_rng(123)
            values = np.array([closure.sample_energy(state, rng) for _ in range(100000)])
            self.assertTrue(np.all((values > 0.0) & (values < 1.0)))

    def test_refuses_enabled_but_undeployed_corrections(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            data = dict(np.load(path, allow_pickle=False))
            data["beta_deployed"] = np.zeros_like(data["beta_deployed"])
            np.savez_compressed(path, **data)
            with self.assertRaisesRegex(ValueError, "no beta is deployed"):
                VariationalClosure(path, corrections_enabled=True)
            VariationalClosure(path, corrections_enabled=False)

    def test_feature_domain_is_counted_not_clipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            closure = VariationalClosure(path)
            features = np.zeros(len(FEATURE_NAMES)); features[0] = 0.7
            state = closure.kernel_state(0.8, 0.5, 1.5, features)
            self.assertTrue(state["out_of_domain"])
            self.assertEqual(closure.out_of_domain_fraction, 1.0)

    def test_joint_angular_parameters_interpolate_inside_deployed_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path, joint=True)
            closure = VariationalClosure(path)
            state = closure.kernel_state(0.9, 0.75, 1.75,
                                         np.zeros(len(FEATURE_NAMES)))
            self.assertTrue(state["joint_deployed"])
            np.testing.assert_allclose(state["joint_parameters"], [0.1, -0.1, 0.3])

    def test_runtime_gate_reports_each_release_failure(self):
        accepted = runtime_gate_status({
            "negative_energy_repairs": 0,
            "out_of_domain_fraction": 0.0009,
            "closure_overhead_fraction": 0.049,
        })
        self.assertTrue(accepted["pass"])
        rejected = runtime_gate_status({
            "negative_energy_repairs": 1,
            "out_of_domain_fraction": 0.001,
            "closure_overhead_fraction": 0.05,
        })
        self.assertFalse(rejected["pass"])
        self.assertEqual(len(rejected["reasons"]), 3)

    def test_variational_loss_loader_has_no_gmm_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dissipation = root / "dissipation"
            dissipation.mkdir()
            table = {"(0.8,2.0)": 0.4}
            (dissipation / "gamma_max_table.json").write_text(json.dumps(table))
            (dissipation / "one_hit_table.json").write_text(json.dumps(table))
            model = FrozenLossModel(root)
            self.assertFalse(hasattr(model, "cond_gmm"))
            self.assertAlmostEqual(model.loss_parameters(0.8, 2.0)["gamma_max"], 0.4)

    def test_variational_simulation_runs_with_loss_tables_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "closure_v2.npz"
            self._write(artifact)
            dissipation = root / "dissipation"
            dissipation.mkdir()
            (dissipation / "gamma_max_table.json").write_text(
                json.dumps({"(0.8,2.0)": 0.4}))
            (dissipation / "one_hit_table.json").write_text(
                json.dumps({"(0.8,2.0)": 0.4}))
            config = {
                "particle": {"AR": 2.0, "radius": 0.5, "mass": 1.0},
                "system": {"kTt": 1.0, "kTr": 1.0, "alpha": 0.8,
                           "phi": 0.01, "domain": [20.0, 20.0, 20.0]},
                "time": {"dt": 0.01, "dtau": 0.1, "t_end": 0.02,
                         "tau_end": None, "equilibration_time": 0.0},
                "flow": {"mode": "hcs", "shear_rate": 0.0},
                "simulation": {"sphere_collision": False, "use_isotropic_eps": True},
                "preprocessing": {"model_root": str(root),
                                  "dissipation": {"beta_a": 1.21, "beta_b": 3.67}},
                "microscopic_closure": {
                    "routing": "variational_v2", "angular": "variational_v2",
                    "artifact": str(artifact), "invariant_corrections": False,
                },
            }
            diagnostics = run_simulation(config, 42, root / "hcs.txt")
            self.assertEqual(diagnostics["routing"], "variational_v2")
            self.assertEqual(diagnostics["negative_energy_repairs"], 0)
            self.assertIsNotNone(diagnostics["runtime_gate"])


if __name__ == "__main__":
    unittest.main()
