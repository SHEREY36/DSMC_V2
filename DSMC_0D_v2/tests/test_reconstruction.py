import unittest

import numpy as np

from dsmc_v2.reconstruction import reconstruct_post_state


class ReconstructionTests(unittest.TestCase):
    def test_conserves_energy_and_angular_momentum(self):
        rng = np.random.default_rng(8)
        v1, v2 = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        u1, u2 = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        w1, w2 = np.array([0.2, 0.1, 0.0]), np.array([0.3, 0.0, 0.2])
        mass, moi = 1.0, 0.5
        etr = 0.5 * mass * np.dot(v2 - v1, v2 - v1)
        er1, er2 = moi * np.dot(w1, w1), moi * np.dot(w2, w2)
        total = etr + er1 + er2
        outcome = np.array([1.0, etr / total, er1 / total, er2 / total,
                            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, -0.2])
        result = reconstruct_post_state(v1, v2, w1, w2, u1, u2,
                                        np.array([1.0, 0.0, 0.0]), outcome,
                                        mass, moi, 2.0, rng)
        self.assertIsNotNone(result)
        self.assertLess(result.energy_error, 1.0e-12)
        self.assertLess(result.angular_momentum_error, 1.0e-12)
        self.assertAlmostEqual(np.dot(result.omega1, u1), 0.0, places=12)
        self.assertAlmostEqual(np.dot(result.omega2, u2), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()

