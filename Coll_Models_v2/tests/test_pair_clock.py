import unittest

import numpy as np

from coll_models_v2.pair_clock import _spline_design


class PairClockTests(unittest.TestCase):
    def test_design_is_finite_and_additive(self):
        rng = np.random.default_rng(9)
        values = rng.normal(size=(20, 7))
        knots = np.tile(np.array([-0.5, 0.0, 0.5]), (7, 1))
        design = _spline_design(values, knots)
        self.assertEqual(design.shape, (20, 1 + 2 * 7 + 3 * 7))
        self.assertTrue(np.isfinite(design).all())


if __name__ == "__main__":
    unittest.main()

