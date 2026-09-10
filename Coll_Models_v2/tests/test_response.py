import unittest

import numpy as np

from coll_models_v2.fit_coefficients import fit_lambda1_coefficients
from dsmc_v2_contracts import FEATURE_NAMES


class ResponseFitTests(unittest.TestCase):
    def test_lambda1_fit_is_centered_and_validated_on_extreme_amplitudes(self):
        center = np.linspace(-0.03, 0.04, len(FEATURE_NAMES))
        truth = np.linspace(-0.8, 0.9, len(FEATURE_NAMES))
        baseline = {
            "ensemble_id": 0,
            "cell_features": dict(zip(FEATURE_NAMES, center)),
            "energy": {"lambda1": 0.7},
            "uncertainty": {"lambda1": {"standard_error": 1.0e-4}},
        }
        nodes = [baseline]
        ensemble = 1
        for feature_index, family in enumerate(FEATURE_NAMES):
            for eta in (-0.5, -0.25, 0.25, 0.5):
                delta = np.zeros(len(FEATURE_NAMES))
                delta[feature_index] = eta
                nodes.append({
                    "ensemble_id": ensemble,
                    "cell_features": dict(zip(FEATURE_NAMES, center + delta)),
                    "energy": {"lambda1": 0.7 + float(truth @ delta)},
                    "uncertainty": {"lambda1": {"standard_error": 1.0e-4}},
                    "excitation": {"family": family, "eta": eta},
                })
                ensemble += 1
        fitted = fit_lambda1_coefficients(nodes)
        np.testing.assert_allclose(fitted["feature_center"], center)
        np.testing.assert_allclose(fitted["beta"], truth, atol=1.0e-10)
        self.assertEqual(fitted["design_rank"], len(FEATURE_NAMES))
        self.assertTrue(fitted["linearity_pass"])
        self.assertLess(fitted["validation_relative_rmse"], 1.0e-10)


if __name__ == "__main__":
    unittest.main()
