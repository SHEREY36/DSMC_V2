import unittest

from coll_models_v2.pipeline import precision_status
from dsmc_v2_contracts import FEATURE_NAMES


def _quantity(value):
    return {"estimate": value, "standard_error": 0.0,
            "ci_low": value, "ci_high": value}


class PipelineQATests(unittest.TestCase):
    def test_preserved_bl_mismatch_is_audit_not_continuation_reason(self):
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
                "cross_section_pass": True,
                "total_loss_compatibility_pass": False,
                "vss_representable": True,
                "score_tail_pass": True,
            },
        }
        passed, reasons = precision_status(result)
        self.assertTrue(passed)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
