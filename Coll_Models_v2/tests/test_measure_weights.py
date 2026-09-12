import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from coll_models_v2.estimate import _run_events, _run_propensity
from coll_models_v2.weights import effective_sample_size, projected_excluded_area
from dsmc_v2_contracts.io import AI, OI, ATTEMPT_DTYPE, OUTCOME_DTYPE


class MeasureWeightTests(unittest.TestCase):
    @staticmethod
    def _run(ar, u1, u2, g):
        attempts = np.zeros(1, dtype=ATTEMPT_DTYPE)
        values = attempts["values"]
        for prefix, vector in (("c1", 0.5 * np.asarray(g)),
                               ("c2", -0.5 * np.asarray(g)),
                               ("u1", np.asarray(u1)), ("u2", np.asarray(u2))):
            for component, axis in enumerate("xyz"):
                values[0, AI[f"{prefix}_{axis}"]] = vector[component]
        return SimpleNamespace(attempts=attempts, metadata={"aspect_ratio": ar, "diameter": 1.0})

    def test_sphere_projected_area_is_pi_d_squared(self):
        area = projected_excluded_area(self._run(1.0, [1, 0, 0], [0, 1, 0], [0, 0, 1]))
        self.assertAlmostEqual(area[0], np.pi)

    def test_crossed_rods_include_triple_product_term(self):
        area = projected_excluded_area(self._run(2.0, [1, 0, 0], [0, 1, 0], [0, 0, 1]))
        self.assertAlmostEqual(area[0], np.pi + 5.0)

    def test_inverse_area_ess_definition(self):
        weight = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(effective_sample_size(weight), 36.0 / 14.0)

    def test_excitation_weight_is_joined_by_attempt_key_not_row_order(self):
        attempts = np.zeros(3, dtype=ATTEMPT_DTYPE)
        attempts["event_id"] = [10, 11, 12]
        attempts["attempt_index"] = [0, 0, 0]
        attempts["hit"] = [1, 0, 1]
        for row in range(3):
            attempts["values"][row, AI["c1_x"]] = 1.0
            attempts["values"][row, AI["c2_x"]] = -1.0
        outcomes = np.zeros(2, dtype=OUTCOME_DTYPE)
        # Deliberately reverse the two hit attempts.
        outcomes["event_id"] = [12, 10]
        outcomes["attempt_index"] = 0
        outcomes["values"][:, OI["e_initial"]] = 5.0
        outcomes["values"][:, OI["et_elastic"]] = 2.0
        outcomes["values"][:, OI["et_inelastic"]] = 2.0
        outcomes["values"][:, OI["ghat_pre_x"]] = 1.0
        outcomes["values"][:, OI["ghat_post_x"]] = 1.0
        run = SimpleNamespace(
            attempts=attempts, outcomes=outcomes,
            metadata={"mass": 1.0, "seed": 4})
        events = _run_events(
            run, measure="collision", attempt_weight=np.array([2.0, 3.0, 5.0]))
        np.testing.assert_allclose(events["weight"], [5.0, 2.0])

    def test_propensity_cache_cannot_alias_fresh_shards_with_same_name_and_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / side / "same_shard" for side in ("training", "holdout")]
            runs = []
            for seed, directory in enumerate(directories, start=1):
                directory.mkdir(parents=True)
                (directory / "attempts_v2.bin").write_bytes(b"same byte count")
                runs.append(SimpleNamespace(
                    directory=directory, attempts=np.zeros(3), metadata={"seed": seed}))
            generated = [np.full(3, 0.1), np.full(3, 0.2)]
            with patch("coll_models_v2.estimate.kinematic_propensity",
                       side_effect=generated) as calculate:
                first = _run_propensity(runs[0], 8, cache=root / "cache")
                second = _run_propensity(runs[1], 8, cache=root / "cache")
            self.assertEqual(calculate.call_count, 2)
            np.testing.assert_array_equal(first, generated[0])
            np.testing.assert_array_equal(second, generated[1])


if __name__ == "__main__":
    unittest.main()
