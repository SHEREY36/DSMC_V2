import unittest
from types import SimpleNamespace

import numpy as np

from coll_models_v2.weights import effective_sample_size, projected_excluded_area
from dsmc_v2_contracts.io import AI, ATTEMPT_DTYPE


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


if __name__ == "__main__":
    unittest.main()
