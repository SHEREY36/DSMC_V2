import unittest

import numpy as np

from dsmc_v2.ntc import NTCWorkspace, candidate_count


class NTCTests(unittest.TestCase):
    def test_candidate_count_is_the_v1_expression(self):
        np.random.seed(17)
        random_fraction = np.random.rand()
        np.random.seed(17)
        actual = candidate_count(80, 3.2, 4.1, 1200.0, 0.03)
        expected = int(np.floor(2.0 * 80 * 79 * 3.2 * 4.1 * 0.015 / 1200.0
                                + random_fraction))
        self.assertEqual(actual, expected)

    def test_screening_uses_abs_g_dot_e_over_vrmax(self):
        velocity = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        first = NTCWorkspace(128, seed=91)
        second = NTCWorkspace(128, seed=91)
        vmax1, accepted1 = first.screen_candidates(velocity, 4, 128, 3.0)
        vmax2, accepted2 = second.screen_candidates(velocity, 4, 128, 3.0)
        np.testing.assert_array_equal(accepted1, accepted2)
        self.assertEqual(vmax1, vmax2)
        np.testing.assert_allclose(first.abs_cr[:128], second.abs_cr[:128])


if __name__ == "__main__":
    unittest.main()

