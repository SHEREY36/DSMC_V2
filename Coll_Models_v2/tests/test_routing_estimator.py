import unittest

import numpy as np

from coll_models_v2.estimate import N_SCALARS, _evaluate
from dsmc_v2_contracts import LEGACY_FEATURE_NAMES


class _KnownBL:
    @staticmethod
    def parameters(alpha, aspect_ratio):
        return {"mean_loss_fraction": 0.5}


class RoutingEstimatorTests(unittest.TestCase):
    def test_recovers_known_dsmc_compatible_derivatives(self):
        sums = np.zeros(N_SCALARS + 4 * len(LEGACY_FEATURE_NAMES))
        sums[:10] = [100, 50, 20, 10, 100, 40, 1, 2, 3, 4]
        expected = np.linspace(-0.3, 0.3, 16)
        lbl = np.linspace(-0.1, 0.1, 16)
        start = N_SCALARS
        sums[start + 16:start + 32] = 20 * (lbl + 0.05)
        sums[start + 32:start + 48] = 10 * (lbl + expected)
        sums[start + 48:start + 64] = 100 * lbl
        metadata = {"alpha": 0.8, "theta": 1.0, "aspect_ratio": 2.0,
                    "proposal_area": 4.0, "collision_cross_section": 2.0}
        result = _evaluate(sums, metadata, _KnownBL())
        np.testing.assert_allclose(result[13:29], expected, atol=1.0e-14)
        self.assertAlmostEqual(result[3], 0.4)
        np.testing.assert_allclose(result[29:45], expected - 0.05, atol=1.0e-14)

    def test_modal_routing_above_one_is_valid(self):
        sums = np.zeros(N_SCALARS + 4 * len(LEGACY_FEATURE_NAMES))
        # F0 = 4*30/(2*(0.5*100)) = 1.2 and FC = 30/20 = 1.5.
        # Both express rotational gain accompanying translational loss.
        sums[:10] = [100, 50, 20, 30, 100, 40, 1, 2, 3, 4]
        metadata = {"alpha": 0.8, "theta": 1.0, "aspect_ratio": 2.0,
                    "proposal_area": 4.0, "collision_cross_section": 2.0}
        result = _evaluate(sums, metadata, _KnownBL())
        self.assertAlmostEqual(result[3], 1.2)
        self.assertAlmostEqual(result[5], 1.5)


if __name__ == "__main__":
    unittest.main()
