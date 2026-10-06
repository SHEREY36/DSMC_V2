import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from coll_models_v2.projections import angular_quantiles, energy_quantile_table
from dsmc_v2.artifact import VariationalClosure
from dsmc_v2.legacy_models import FrozenLossModel
from dsmc_v2.non_gaussian import NonGaussianDiagnostics
from dsmc_v2.simulation import run_simulation, runtime_gate_status
from dsmc_v2.state import ParticleState
from dsmc_v2_contracts import FEATURE_NAMES, ONE_SIDED_FEATURE_NAMES


class VariationalArtifactTests(unittest.TestCase):
    @staticmethod
    def _write(path, joint=False):
        coordinates = np.array([[a, t, r] for a in (0.8, 1.0)
                                 for t in (0.1, 3.0) for r in (1.5, 2.0)])
        probability = np.linspace(0.0, 1.0, 513)
        # (lambda1, lambda2, lambda3, lambda4); a real memory so the sampler's
        # a-axis is exercised rather than collapsing to a point.
        ep = np.zeros((len(coordinates), 4))
        ep[:, 2] = 5.0
        a_grid = np.array([np.linspace(row[0] - 1.0, row[0] + row[2] + 1.0, 65)
                           for row in ep])
        ap = np.zeros((len(coordinates), 2))
        beta = np.zeros((len(coordinates), len(FEATURE_NAMES)))
        beta[:, 0] = 0.2
        np.savez_compressed(
            path, schema_version=np.array("2.3.0"),
            artifact_type=np.array("bl_variational_closure"),
            feature_names=np.array(FEATURE_NAMES), surface_coordinates=coordinates,
            p_exch=np.full(len(coordinates), 0.4), energy_parameters=ep,
            angular_parameters=ap, quantile_probability=probability,
            energy_a_grid=a_grid, kernel_form=np.array("sinkhorn_bridge_v2"),
            energy_mean_loss=np.zeros(len(coordinates)),
            energy_anchor=np.zeros((len(coordinates), 2)),
            xi_grid=np.geomspace(0.05, 40.0, 32),
            xi_enhancement=np.ones((len(coordinates), 32)),
            energy_quantiles=np.array([
                energy_quantile_table(row[2], row[1], grid, probability)
                for row, grid in zip(ep, a_grid)]),
            angular_quantiles=np.array([angular_quantiles(row, probability) for row in ap]),
            beta_coordinates=coordinates, beta=beta, beta_se=np.zeros_like(beta),
            beta_deployed=beta != 0.0,
            correction_trust_amplitude=np.array(0.25),
            feature_lower=np.full(len(FEATURE_NAMES), -0.5),
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
            self.assertEqual(closure.energy_interpolation,
                             "node_first_quantile_interpolation_v1")
            self.assertAlmostEqual(state["p_exch"], 0.4)
            # The retired flow-moment response must not move the base law.
            self.assertAlmostEqual(state["energy_parameters"][0], 0.0)
            bare = closure.kernel_state(0.9, 0.75, 1.75)
            np.testing.assert_allclose(bare["energy_parameters"], state["energy_parameters"])
            rng = np.random.default_rng(123)
            values = np.array([closure.sample_energy(state, 0.5, 0.0, rng)
                               for _ in range(100000)])
            self.assertTrue(np.all((values > 0.0) & (values < 1.0)))
            self.assertAlmostEqual(
                np.mean(values), closure.mean_energy(state, 0.5, 0.0), places=3)
            # Memory must actually bite: a larger incoming share must push the
            # outgoing share up, or lambda3 is being dropped again.
            low = np.mean([closure.sample_energy(state, 0.1, 0.0, rng)
                           for _ in range(20000)])
            high = np.mean([closure.sample_energy(state, 0.9, 0.0, rng)
                            for _ in range(20000)])
            self.assertGreater(high - low, 0.05)

    def test_invariant_response_is_retired(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            with self.assertRaisesRegex(ValueError, "retired"):
                VariationalClosure(path, corrections_enabled=True)
            # An artifact with no response fields at all loads.
            data = dict(np.load(path, allow_pickle=False))
            for key in [k for k in data if k.startswith("beta") or k.startswith("correction")]:
                data.pop(key)
            np.savez_compressed(path, **data)
            closure = VariationalClosure(path)
            state = closure.kernel_state(0.8, 0.1, 1.5)
            self.assertFalse(state["energy_corrected"])

    def test_packed_adaptive_energy_tables_match_rectangular_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            rectangular_path = Path(temporary) / "rectangular.npz"
            packed_path = Path(temporary) / "packed.npz"
            self._write(rectangular_path)
            data = dict(np.load(rectangular_path, allow_pickle=False))
            grids = data["energy_a_grid"]
            tables = data["energy_quantiles"]
            offsets = np.r_[0, np.cumsum([len(grid) for grid in grids])]
            data["energy_a_grid"] = np.concatenate(list(grids))
            data["energy_quantiles"] = np.concatenate(list(tables), axis=0)
            data["energy_a_offsets"] = offsets
            np.savez_compressed(packed_path, **data)

            rectangular = VariationalClosure(
                rectangular_path, corrections_enabled=False)
            packed = VariationalClosure(packed_path, corrections_enabled=False)
            features = np.zeros(len(FEATURE_NAMES))
            state_r = rectangular.kernel_state(0.9, 0.75, 1.75, features)
            state_p = packed.kernel_state(0.9, 0.75, 1.75, features)
            self.assertEqual(packed.energy_table_layout, "packed_adaptive_v1")
            for z_in in (0.05, 0.4, 0.95):
                self.assertAlmostEqual(
                    packed.mean_energy(state_p, z_in, 0.0),
                    rectangular.mean_energy(state_r, z_in, 0.0), places=14)

    def test_physical_interpolation_preserves_exact_grid_planes(self):
        """An exact sampled alpha must not borrow a neighbouring alpha plane.

        Generic Delaunay interpolation through a Cartesian grid can select a
        diagonal simplex that does so, even though a tensor-product stencil is
        available and has the physically unambiguous answer.
        """
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            closure = VariationalClosure(path, corrections_enabled=False)
            state = closure.kernel_state(0.8, 0.75, 1.75,
                                         np.zeros(len(FEATURE_NAMES)))
            indices = state["energy_vertex_indices"]
            np.testing.assert_allclose(closure.coordinates[indices, 0], 0.8)
            self.assertAlmostEqual(float(np.sum(state["energy_vertex_weights"])), 1.0)

    def test_energy_sampler_evaluates_nodes_before_mixing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            closure = VariationalClosure(path, corrections_enabled=False)
            state = closure.kernel_state(0.9, 0.75, 1.75,
                                         np.zeros(len(FEATURE_NAMES)))
            z_in = 0.37
            rows = []
            for index in state["energy_vertex_indices"]:
                lambda1, _, lambda3, lambda4 = closure.energy_parameters[index]
                a = lambda1 + lambda3 * z_in + lambda4 * 0.0
                grid = closure.energy_a_grid[index]
                upper = int(np.searchsorted(grid, a).clip(1, len(grid) - 1))
                lower = upper - 1
                blend = (a - grid[lower]) / (grid[upper] - grid[lower])
                table = closure.energy_tables[index]
                rows.append((1.0 - blend) * table[lower] + blend * table[upper])
            expected_row = np.tensordot(state["energy_vertex_weights"], rows,
                                        axes=(0, 0))
            expected_mean = np.trapezoid(expected_row, closure.probability)
            self.assertAlmostEqual(closure.mean_energy(state, z_in, 0.0),
                                   expected_mean, places=13)

    def test_bounded_logit_kernel_loads_and_uses_nonlinear_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            data = dict(np.load(path, allow_pickle=False))
            count = len(data["surface_coordinates"])
            coefficients = np.tile([3.0, -1.0, 0.5], (count, 1))
            center = np.zeros(count)
            scale = np.ones(count)
            z = np.linspace(1.0e-9, 1.0 - 1.0e-9, 2049)
            x = np.tanh(np.log(z / (1.0 - z)) / 2.0)
            shift = coefficients[0, 0] * x + coefficients[0, 1] * x**2 \
                + coefficients[0, 2] * x**3
            ep = np.zeros((count, 4)); ep[:, 1] = -1.1
            a_grid = np.tile(np.linspace(shift.min() - 0.1, shift.max() + 0.1, 129),
                             (count, 1))
            probability = data["quantile_probability"]
            tables = np.array([
                energy_quantile_table(0.0, row[1], grid, probability,
                                      kernel_form="conditional_logit_cubic_v3")
                for row, grid in zip(ep, a_grid)])
            data.update(
                p_exch=np.full(count, -0.05),
                energy_parameters=ep,
                energy_a_grid=a_grid,
                energy_quantiles=tables,
                kernel_form=np.array("conditional_logit_cubic_v3"),
                energy_kernel_forms=np.full(count, "conditional_logit_cubic_v3"),
                energy_memory_coefficients=coefficients,
                energy_memory_center=center,
                energy_memory_scale=scale,
            )
            np.savez_compressed(path, **data)
            closure = VariationalClosure(path, corrections_enabled=False)
            state = closure.kernel_state(0.8, 0.1, 1.5,
                                         np.zeros(len(FEATURE_NAMES)))
            self.assertLess(state["p_exch"], 0.0)
            low = closure.mean_energy(state, 0.02, 0.0)
            high = closure.mean_energy(state, 0.98, 0.0)
            self.assertGreater(high - low, 0.1)

    def test_refuses_an_artifact_without_the_collision_measure(self):
        """Fail closed. A missing enhancement silently reverts the runtime to
        the orientation-isotropic proposal ensemble -- the exact bug the table
        exists to remove -- and would do it with a plausible-looking answer."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            data = dict(np.load(path, allow_pickle=False))
            del data["xi_enhancement"]
            np.savez_compressed(path, **data)
            with self.assertRaisesRegex(ValueError, "collision-measure enhancement"):
                VariationalClosure(path)

    def test_enhancement_must_be_per_node(self):
        """One global curve is wrong: it depends on aspect ratio and on theta."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            data = dict(np.load(path, allow_pickle=False))
            data["xi_enhancement"] = np.ones(32)
            np.savez_compressed(path, **data)
            with self.assertRaisesRegex(ValueError, "xi_enhancement must be"):
                VariationalClosure(path)

    def test_feature_domain_is_inactive_when_corrections_are_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "closure_v2.npz"
            self._write(path)
            closure = VariationalClosure(path, corrections_enabled=False)
            features = np.zeros(len(FEATURE_NAMES)); features[0] = 0.7
            state = closure.kernel_state(0.8, 0.5, 1.5, features)
            self.assertFalse(state["out_of_domain"])
            self.assertFalse(state["correction_fallback"])
            self.assertEqual(closure.out_of_domain_fraction, 0.0)

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
            "correction_fallback_fraction_in_evaluation_window": 0.009,
            "closure_overhead_fraction": 0.149,
        })
        self.assertTrue(accepted["pass"])
        rejected = runtime_gate_status({
            "negative_energy_repairs": 1,
            "correction_fallback_fraction_in_evaluation_window": 0.01,
            "closure_overhead_fraction": 0.15,
            "energy_axis_clamps": 1,
            "energy_monotonic_repairs": 1,
            "maximum_bulk_to_thermal_temperature_ratio": 1.0e-12,
        })
        self.assertFalse(rejected["pass"])
        self.assertEqual(len(rejected["reasons"]), 6)
        adaptive = runtime_gate_status({
            "negative_energy_repairs": 0,
            "correction_fallback_fraction_in_evaluation_window": 0.9,
            "correction_fallback_policy": "base_law",
            "closure_overhead_fraction": 0.01,
        }, correction_fallback_gate="adaptive_base_law")
        self.assertTrue(adaptive["pass"])
        self.assertFalse(adaptive["correction_support_pass"])
        self.assertEqual(adaptive["reasons"], [])
        unsafe_fallback = runtime_gate_status({
            "negative_energy_repairs": 0,
            "correction_fallback_fraction_in_evaluation_window": 0.9,
            "correction_fallback_policy": "clip",
            "closure_overhead_fraction": 0.01,
        }, correction_fallback_gate="adaptive_base_law")
        self.assertFalse(unsafe_fallback["pass"])
        self.assertIn("correction_fallback_policy_not_base_law",
                      unsafe_fallback["reasons"])
        with self.assertRaises(ValueError):
            runtime_gate_status({
                "negative_energy_repairs": 0,
                "closure_overhead_fraction": 0.0,
            }, correction_fallback_gate="clip")

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
                "simulation": {
                    "sphere_collision": False, "use_isotropic_eps": True,
                    "orientation_integrator": "symmetric_midpoint_v1",
                },
                "preprocessing": {"model_root": str(root),
                                  "dissipation": {"beta_a": 1.21, "beta_b": 3.67}},
                "microscopic_closure": {
                    "routing": "variational_v2", "angular": "variational_v2",
                    "artifact": str(artifact), "invariant_corrections": False,
                },
            }
            axis_steps = []
            sample_taus = []
            advance_axes = ParticleState.advance_axes
            maybe_sample = NonGaussianDiagnostics.maybe_sample

            def tracked_advance_axes(state, step):
                axis_steps.append(float(step))
                return advance_axes(state, step)

            def tracked_maybe_sample(diagnostic, time, tau, state):
                sample_taus.append(float(tau))
                return maybe_sample(diagnostic, time, tau, state)

            with (patch.object(
                    ParticleState, "advance_axes", new=tracked_advance_axes),
                  patch.object(NonGaussianDiagnostics, "maybe_sample",
                               new=tracked_maybe_sample)):
                diagnostics = run_simulation(config, 42, root / "hcs.txt")
            self.assertEqual(diagnostics["routing"], "variational_v2")
            self.assertEqual(diagnostics["orientation_integrator"],
                             "symmetric_midpoint_v1")
            self.assertGreaterEqual(len(axis_steps), 2)
            self.assertEqual(len(axis_steps) % 2, 0)
            self.assertTrue(all(np.isclose(step, 0.005) for step in axis_steps))
            self.assertGreaterEqual(len(sample_taus), 2)
            self.assertAlmostEqual(sample_taus[-1], diagnostics["cpp"])
            self.assertEqual(diagnostics["energy_interpolation"],
                             "node_first_quantile_interpolation_v1")
            self.assertEqual(diagnostics["negative_energy_repairs"], 0)
            self.assertAlmostEqual(
                diagnostics["ntc"]["initial_vrmax_temperature_bound"],
                1.0 + 2.0 / 3.0)
            self.assertIsNotNone(diagnostics["runtime_gate"])


if __name__ == "__main__":
    unittest.main()
