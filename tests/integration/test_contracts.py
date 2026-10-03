import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import (
    ATTEMPT_DTYPE, OUTCOME_DTYPE, FEATURE_NAMES, ONE_SIDED_FEATURE_NAMES,
    cell_features, cell_features_with_domain, cell_invariants, load_run,
    validate_run,
)
from dsmc_v2_contracts.io import OI
from dsmc_v2_contracts.features import (
    _normalised_state, _particle_moments, _u_contraction,
    _u_dot, _u_v_square_contraction, _u_v_square_dot,
)


class ContractTests(unittest.TestCase):
    def test_record_sizes_are_frozen(self):
        self.assertEqual(ATTEMPT_DTYPE.itemsize, 200)
        self.assertEqual(OUTCOME_DTYPE.itemsize, 552)
        self.assertIn("ensemble_id", ATTEMPT_DTYPE.names)
        self.assertNotIn("reserved", ATTEMPT_DTYPE.names)
        self.assertEqual(len(FEATURE_NAMES), 14)

    def test_synthetic_run_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {
                "schema_version": "2.1.0", "nsamples": 1, "seed": 1,
                "alpha": 0.8, "theta": 1.0, "aspect_ratio": 1.0,
                "velocity_scale": np.sqrt(2.0), "omega_scale": np.sqrt(2.0),
                "proposal_area": 4.0, "byte_order": "little", "mass": 1.0,
                "moi_perpendicular": 1.0,
                "attempt_record_bytes": 200, "outcome_record_bytes": 552,
            }
            (root / "metadata_v2.json").write_text(json.dumps(metadata))
            attempt = np.zeros(1, dtype=ATTEMPT_DTYPE)
            attempt["event_id"], attempt["attempt_index"], attempt["hit"] = 1, 1, 1
            av = attempt["values"][0]
            av[0:3], av[3:6] = [-1, 0, 0], [1, 0, 0]
            av[12:15], av[15:18] = [0, 0, 1], [0, 1, 0]
            attempt.tofile(root / "attempts_v2.bin")
            outcome = np.zeros(1, dtype=OUTCOME_DTYPE)
            outcome["event_id"], outcome["attempt_index"], outcome["n_contact"] = 1, 1, 1
            ov = outcome["values"][0]
            ov[0:18] = av[:18]
            ov[18:36] = av[:18]
            ov[39:42], ov[42:45] = [1, 0, 0], [1, 0, 0]
            ov[47:53] = [2.0, 0.0, 1.8, 0.0, 0.0, 2.0]
            ov[OI["delta_tr"]], ov[OI["delta_total"]] = 0.2, 0.2
            ov[59:62], ov[62:65] = [1, 0, 0], [1, 0, 0]
            outcome.tofile(root / "outcomes_v2.bin")
            run = load_run(root)
            self.assertEqual(run.metadata["schema_adapter"], "read_only_2.1_to_2.2")
            self.assertEqual(int(run.attempts["ensemble_id"][0]), 0)
            qa = validate_run(run)
            self.assertEqual(qa["status"], "pass", qa)

    def test_isotropic_features_are_small(self):
        rng = np.random.default_rng(7)
        n = 50000
        velocity = rng.normal(size=(n, 3))
        axis = rng.normal(size=(n, 3)); axis /= np.linalg.norm(axis, axis=1)[:, None]
        omega = rng.normal(size=(n, 3))
        omega -= np.einsum("ni,ni->n", omega, axis)[:, None] * axis
        values = cell_features(velocity, omega, axis)
        self.assertLess(np.max(np.abs(values)), 0.04)
        self.assertTrue(np.isfinite(values).all())

    def test_domain_features_do_not_misclassify_negative_u_statistic_noise(self):
        rng = np.random.default_rng(193)
        velocity = rng.normal(size=(32, 3))
        axis = rng.normal(size=(32, 3))
        axis /= np.linalg.norm(axis, axis=1)[:, None]
        omega = rng.normal(size=(32, 3))
        omega -= np.einsum("ni,ni->n", omega, axis)[:, None] * axis
        raw, domain = cell_features_with_domain(velocity, omega, axis)
        one_sided = np.array([
            FEATURE_NAMES.index(name) for name in ONE_SIDED_FEATURE_NAMES])
        signed = np.array([
            i for i in range(len(FEATURE_NAMES)) if i not in set(one_sided)])
        self.assertTrue(np.all(domain[one_sided] >= 0.0))
        np.testing.assert_allclose(domain[signed], raw[signed])
        # The correction retains the unbiased estimator rather than silently
        # replacing it with the support-test statistic.
        self.assertGreater(np.max(np.abs(domain[one_sided] - raw[one_sided])), 0.0)

    def test_aggregate_feature_path_is_algebraically_identical(self):
        """The allocation-free path must preserve every deployed invariant."""
        rng = np.random.default_rng(194)
        for sphere in (False, True):
            velocity = rng.normal(size=(37, 3))
            axis = rng.normal(size=(37, 3))
            axis /= np.linalg.norm(axis, axis=1)[:, None]
            omega = rng.normal(size=(37, 3))
            omega -= np.einsum("ni,ni->n", omega, axis)[:, None] * axis
            c, w, u = _normalised_state(velocity, omega, axis, 1.0, 1.0)
            moments = _particle_moments(c, w, u)
            if sphere:
                moments["rt"] -= moments["qq"]
                moments["qq"][:] = 0.0
                moments["acu"][:] = 0.0
            x, y = moments["x"], moments["y"]
            expected = np.array([
                (4.0 / 15.0) * np.mean(x * x) - 1.0,
                0.5 * np.mean(y * y) - 1.0,
                (2.0 / 3.0) * np.mean(x * y) - 1.0,
                np.mean(moments["acu"]),
                _u_v_square_contraction(moments["pi"])[0] / 8.0,
                _u_v_square_contraction(moments["qq"])[0] / 8.0,
                _u_v_square_contraction(moments["rt"])[0] / 8.0,
                _u_contraction(moments["pi"], moments["qq"]) / 4.0,
                _u_contraction(moments["pi"], moments["rt"]) / 4.0,
                _u_contraction(moments["qq"], moments["rt"]) / 4.0,
                _u_v_square_dot(moments["qtr"])[0],
                _u_v_square_dot(moments["qrot"])[0],
                _u_dot(moments["qtr"], moments["qrot"]),
                _u_v_square_dot(moments["w"])[0],
            ])
            expected_diagnostics = np.array([
                _u_dot(moments["acw"][:, None], moments["acw"][:, None]),
                _u_dot(moments["vx"], moments["vx"]),
            ])
            actual, actual_diagnostics = cell_invariants(
                velocity, omega, axis, sphere=sphere)
            np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(
                actual_diagnostics, expected_diagnostics, rtol=2e-14, atol=2e-14)


if __name__ == "__main__":
    unittest.main()
