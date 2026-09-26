import unittest

import numpy as np

from coll_models_v2.fit_coefficients import (
    CORRECTION_PARAMETERS,
    fit_correction_coefficients,
    fit_lambda1_coefficients,
)
from coll_models_v2.artifact import _fit_coefficient_rows
from coll_models_v2.excitation import EXCITATION_FAMILIES
from dsmc_v2_contracts import FEATURE_NAMES


class ResponseFitTests(unittest.TestCase):
    @staticmethod
    def _linear_response_node_group(sentinel_pass=True):
        center = np.zeros(len(FEATURE_NAMES))
        baseline = {
            "alpha": 0.8, "theta": 1.0, "aspect_ratio": 2.0,
            "ensemble_id": 0,
            "cell_features": dict(zip(FEATURE_NAMES, center)),
            "energy": {"lambda1": 0.0, "lambda2": 0.0,
                       "lambda3": 0.0, "lambda4": 0.0},
            "angular": {"eta1": 0.0, "eta2": 0.0},
            "uncertainty": {name: {"standard_error": 1.0e-5}
                            for _, name in CORRECTION_PARAMETERS},
        }
        nodes = [baseline]
        for family in EXCITATION_FAMILIES:
            feature_index = FEATURE_NAMES.index(family.removesuffix("_opposed"))
            for eta in (-0.5, -0.25, 0.25, 0.5):
                delta = np.zeros(len(FEATURE_NAMES))
                delta[feature_index] = eta
                nodes.append({
                    "alpha": 0.8, "theta": 1.0, "aspect_ratio": 2.0,
                    "ensemble_id": len(nodes),
                    "cell_features": dict(zip(FEATURE_NAMES, delta)),
                    "energy": {"lambda1": 0.1 * eta,
                               "lambda2": -0.1 * eta,
                               "lambda3": 0.2 * eta,
                               "lambda4": -0.2 * eta},
                    "angular": {"eta1": 0.15 * eta,
                                "eta2": -0.12 * eta},
                    "uncertainty": {name: {"standard_error": 1.0e-5}
                                    for _, name in CORRECTION_PARAMETERS},
                    "excitation": {"family": family, "eta": eta},
                    "qa": {"sentinel_pass": (
                        sentinel_pass or not np.isclose(abs(eta), 0.25))},
                })
        return nodes

    def test_angular_evidence_policy_holds_back_every_energy_row(self):
        rows = _fit_coefficient_rows(
            self._linear_response_node_group(),
            release_policy="validated-angular-only-v1")
        self.assertEqual(len(rows), 1)
        deployed = np.asarray(rows[0]["beta_deployed"], dtype=bool)
        self.assertFalse(np.any(deployed[:4]))
        self.assertTrue(np.all(deployed[4:]))
        self.assertTrue(rows[0]["angular_release"])
        self.assertEqual(rows[0]["trust_amplitude"], 0.5)

    def test_angular_evidence_policy_fails_closed_on_training_sentinel(self):
        rows = _fit_coefficient_rows(
            self._linear_response_node_group(sentinel_pass=False),
            release_policy="validated-angular-only-v1")
        self.assertFalse(np.any(np.asarray(rows[0]["beta_deployed"])))
        self.assertFalse(rows[0]["angular_release"])
        self.assertEqual(rows[0]["trust_amplitude"], 0.25)

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
