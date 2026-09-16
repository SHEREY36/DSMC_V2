import unittest

import numpy as np

from coll_models_v2.fit_coefficients import (
    CORRECTION_PARAMETERS,
    fit_correction_coefficients,
    fit_lambda1_coefficients,
)
from dsmc_v2_contracts import FEATURE_NAMES


class ResponseFitTests(unittest.TestCase):
    def test_multivariate_fit_recovers_all_natural_parameter_responses(self):
        center = np.zeros(len(FEATURE_NAMES))
        truth = np.vstack([
            np.linspace(0.1 * (index + 1), 0.2 * (index + 1), len(FEATURE_NAMES))
            for index in range(len(CORRECTION_PARAMETERS))
        ])
        baseline = {
            "alpha": 0.9, "ensemble_id": 0,
            "cell_features": dict(zip(FEATURE_NAMES, center)),
            "energy": {name: 0.1 * index for index, (section, name)
                       in enumerate(CORRECTION_PARAMETERS) if section == "energy"},
            "angular": {name: 0.1 * index for index, (section, name)
                        in enumerate(CORRECTION_PARAMETERS) if section == "angular"},
            "uncertainty": {name: {"standard_error": 1.0e-5}
                            for _, name in CORRECTION_PARAMETERS},
        }
        nodes = [baseline]
        ensemble = 1
        for feature_index, family in enumerate(FEATURE_NAMES):
            for eta in (-0.5, -0.25, 0.25, 0.5):
                delta = np.zeros(len(FEATURE_NAMES)); delta[feature_index] = eta
                node = {
                    "alpha": 0.9, "ensemble_id": ensemble,
                    "cell_features": dict(zip(FEATURE_NAMES, delta)),
                    "energy": {}, "angular": {},
                    "uncertainty": {name: {"standard_error": 1.0e-5}
                                    for _, name in CORRECTION_PARAMETERS},
                    "excitation": {"family": family, "eta": eta},
                }
                for output_index, (section, name) in enumerate(CORRECTION_PARAMETERS):
                    node[section][name] = baseline[section][name] + truth[output_index] @ delta
                nodes.append(node); ensemble += 1
        fitted = fit_correction_coefficients(nodes)
        np.testing.assert_allclose(fitted["beta"], truth, atol=1.0e-10)
        self.assertTrue(fitted["linearity_pass"])
        self.assertTrue(np.all(fitted["beta_deployed"]))

    def test_elastic_fit_enforces_detailed_balance_rows(self):
        center = np.zeros(len(FEATURE_NAMES))
        baseline = {
            "alpha": 1.0, "ensemble_id": 0,
            "cell_features": dict(zip(FEATURE_NAMES, center)),
            "energy": {"lambda1": 0.0, "lambda2": 0.0, "lambda3": 2.0,
                       "lambda4": 0.0},
            "angular": {"eta1": 0.0, "eta2": 0.0},
            "uncertainty": {name: {"standard_error": 1.0e-5}
                            for _, name in CORRECTION_PARAMETERS},
        }
        nodes = [baseline]
        for feature_index, family in enumerate(FEATURE_NAMES):
            for eta in (-0.5, -0.25, 0.25, 0.5):
                delta = np.zeros(len(FEATURE_NAMES)); delta[feature_index] = eta
                nodes.append({
                    "alpha": 1.0, "ensemble_id": len(nodes),
                    "cell_features": dict(zip(FEATURE_NAMES, delta)),
                    "energy": {"lambda1": 0.0, "lambda2": 0.0,
                               "lambda3": 2.0 + eta, "lambda4": 0.0},
                    "angular": {"eta1": eta, "eta2": -eta},
                    "uncertainty": {name: {"standard_error": 1.0e-5}
                                    for _, name in CORRECTION_PARAMETERS},
                    "excitation": {"family": family, "eta": eta},
                })
        fitted = fit_correction_coefficients(nodes)
        beta = np.asarray(fitted["beta"])
        deployed = np.asarray(fitted["beta_deployed"])
        np.testing.assert_array_equal(beta[[0, 1, 3]], 0.0)
        self.assertFalse(np.any(deployed[[0, 1, 3]]))

    def test_unresolved_parameter_response_is_not_deployed(self):
        center = np.zeros(len(FEATURE_NAMES))
        baseline = {
            "alpha": 0.8, "ensemble_id": 0,
            "cell_features": dict(zip(FEATURE_NAMES, center)),
            "energy": {"lambda1": 0.2, "lambda2": -0.1,
                       "lambda3": 0.3, "lambda4": 0.1},
            "angular": {"eta1": 0.2, "eta2": 0.0},
            "uncertainty": {name: {"standard_error": 0.01}
                            for _, name in CORRECTION_PARAMETERS},
        }
        nodes = [baseline]
        for feature_index, family in enumerate(FEATURE_NAMES):
            for eta in (-0.5, -0.25, 0.25, 0.5):
                delta = np.zeros(len(FEATURE_NAMES))
                delta[feature_index] = eta
                node = {
                    "alpha": 0.8, "ensemble_id": len(nodes),
                    "cell_features": dict(zip(FEATURE_NAMES, delta)),
                    "energy": {}, "angular": {},
                    "uncertainty": {name: {"standard_error": 0.01}
                                    for _, name in CORRECTION_PARAMETERS},
                    "excitation": {"family": family, "eta": eta},
                }
                for section, name in CORRECTION_PARAMETERS:
                    # Every parameter except eta2 has a clearly resolved
                    # linear response. eta2 stays below three standard errors.
                    slope = 0.2 if name != "eta2" else 0.005
                    node[section][name] = baseline[section][name] + slope * eta
                nodes.append(node)
        fitted = fit_correction_coefficients(nodes)
        eta2 = [name for _, name in CORRECTION_PARAMETERS].index("eta2")
        np.testing.assert_array_equal(np.asarray(fitted["beta"])[eta2], 0.0)
        self.assertFalse(np.any(np.asarray(fitted["beta_deployed"])[eta2]))
        self.assertTrue(
            fitted["parameter_fits"]["eta2"]["immaterial_response_suppressed"])
        self.assertTrue(fitted["linearity_pass"])

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
