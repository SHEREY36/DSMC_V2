import unittest

from coll_models_v2.pipeline import (
    DIRECT_RESPONSE_PRECISION_CONTRACT,
    direct_response_precision_status,
    precision_status,
)
from dsmc_v2_contracts import FEATURE_NAMES


def _quantity(value):
    return {"estimate": value, "standard_error": 0.0,
            "ci_low": value, "ci_high": value}


class PipelineQATests(unittest.TestCase):
    @staticmethod
    def _continuous_result():
        qa = {name: True for name in (
            "propensity_pass", "proposal_balance_pass", "ess_pass",
            "energy_projection_pass", "angular_projection_pass",
            "model_form_pass", "incoming_partition_pass", "elastic_pass")}
        uncertainty = {name: {
            "standard_error": 0.1, "ci_low": -0.2, "ci_high": 0.2,
        } for name in ("lambda1", "lambda2", "lambda3", "reset_mean",
                       "eta1", "eta2")}
        return {
            "ensemble_id": 17,
            "energy": {"kernel_form": "sinkhorn_bridge_v2"},
            "qa": qa,
            "uncertainty": uncertainty,
            "cell_features": {name: 0.5 for name in FEATURE_NAMES},
        }

    def test_preserved_clock_and_bl_mismatches_are_audit_only(self):
        quantities = {
            "sigma_ctc": _quantity(1.0),
            "F_C": _quantity(1.2),
            "B2": _quantity(0.8),
        }
        for name in FEATURE_NAMES:
            quantities[f"beta_ctc_{name}"] = _quantity(0.0)
        result = {
            "alpha": 0.8,
            "theta": 0.5,
            "quantities": quantities,
            "qa": {
                "cross_section_pass": False,
                "total_loss_compatibility_pass": False,
                "vss_representable": True,
                "score_tail_pass": True,
            },
        }
        passed, reasons = precision_status(result)
        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_affine_memory_diagnostic_does_not_veto_continuous_kernel(self):
        qa = {name: True for name in (
            "propensity_pass", "proposal_balance_pass", "ess_pass",
            "energy_projection_pass", "angular_projection_pass",
            "model_form_pass", "incoming_partition_pass", "elastic_pass")}
        qa["memory_diagnostic_pass"] = False
        uncertainty = {name: _quantity(0.0) for name in
                       ("lambda1", "lambda2", "lambda3", "lambda5", "lambda6",
                        "eta1", "eta2")}
        result = {
            "ensemble_id": 0,
            "energy": {"kernel_form": "conditional_logit_cubic_v3"},
            "qa": qa,
            "uncertainty": uncertainty,
        }
        passed, reasons = precision_status(result)
        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_direct_replay_index_is_not_treated_as_excitation_amplitude(self):
        result = self._continuous_result()
        excitation_pass, excitation_reasons = precision_status(result)
        direct_pass, direct_reasons = direct_response_precision_status(result)
        self.assertFalse(excitation_pass)
        self.assertIn("lambda1_contribution_precision", excitation_reasons)
        self.assertTrue(direct_pass)
        self.assertEqual(direct_reasons, [])
        self.assertEqual(
            DIRECT_RESPONSE_PRECISION_CONTRACT,
            "direct-pointwise-uncertainty-v1")

    def test_direct_response_rejects_invalid_uncertainty(self):
        result = self._continuous_result()
        result["uncertainty"]["eta2"]["standard_error"] = float("nan")
        passed, reasons = direct_response_precision_status(result)
        self.assertFalse(passed)
        self.assertIn("eta2_uncertainty_invalid", reasons)


if __name__ == "__main__":
    unittest.main()
