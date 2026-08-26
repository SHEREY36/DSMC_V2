import unittest

import numpy as np

from coll_models_v2.vss import B2_MAX, alpha_eff_from_b2, legendre, sample_vss_cosine, vss_rank2_moment


class VSSTests(unittest.TestCase):
    def test_forward_inverse(self):
        for target in (0.2, 0.6, 1.0, B2_MAX):
            exponent = alpha_eff_from_b2(target)
            self.assertGreaterEqual(exponent, np.sqrt(2.0) - 1.0e-10)
            self.assertAlmostEqual(vss_rank2_moment(exponent), target, places=11)

    def test_monte_carlo_rank2(self):
        rng = np.random.default_rng(11)
        exponent = 6.8
        cosine = np.array([sample_vss_cosine(exponent, rng) for _ in range(150000)])
        measured = np.mean(1.0 - legendre(2, cosine))
        self.assertAlmostEqual(measured, vss_rank2_moment(exponent), delta=0.006)

    def test_out_of_family_fails_without_clipping(self):
        with self.assertRaises(ValueError):
            alpha_eff_from_b2(B2_MAX + 0.01)


if __name__ == "__main__":
    unittest.main()

