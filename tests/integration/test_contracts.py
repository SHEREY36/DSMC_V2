import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsmc_v2_contracts import ATTEMPT_DTYPE, OUTCOME_DTYPE, cell_features, load_run, validate_run
from dsmc_v2_contracts.io import OI


class ContractTests(unittest.TestCase):
    def test_record_sizes_are_frozen(self):
        self.assertEqual(ATTEMPT_DTYPE.itemsize, 200)
        self.assertEqual(OUTCOME_DTYPE.itemsize, 552)

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
            qa = validate_run(load_run(root))
            self.assertEqual(qa["status"], "pass", qa)

    def test_isotropic_features_are_small(self):
        rng = np.random.default_rng(7)
        n = 50000
        velocity = rng.normal(size=(n, 3))
        axis = rng.normal(size=(n, 3)); axis /= np.linalg.norm(axis, axis=1)[:, None]
        omega = rng.normal(size=(n, 3))
        omega -= np.einsum("ni,ni->n", omega, axis)[:, None] * axis
        values = cell_features(velocity, omega, axis)
        self.assertLess(np.max(np.abs(values[:3])), 0.04)
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
