import json
import tempfile
import unittest
from pathlib import Path

from coll_models_v2.artifact import _load_node_estimates
from coll_models_v2.estimate import NODE_ESTIMATE_CONTRACT


class ArtifactInputTests(unittest.TestCase):
    def test_precomputed_node_must_cover_exact_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard1, shard2 = root / "pilot", root / "production"
            shard1.mkdir(); shard2.mkdir()
            estimate = {
                "alpha": 0.5, "theta": 0.1, "aspect_ratio": 1.1,
                "source_runs": [str(shard1), str(shard2)],
                "estimator_contract": NODE_ESTIMATE_CONTRACT,
                "cell_features": {}, "incoming_law": {}, "incoming_law_energy": {},
                "qa": {"precision_pass": True},
            }
            (root / "alpha_0.500_theta_0.100_AR_1.100.json").write_text(
                json.dumps(estimate))
            groups = {(0.5, 0.1, 1.1): [shard1, shard2]}
            self.assertEqual(len(_load_node_estimates(root, groups)), 1)
            groups[(0.5, 0.1, 1.1)].append(root / "continuation")
            with self.assertRaisesRegex(ValueError, "stale node estimate"):
                _load_node_estimates(root, groups)

    def test_reweighted_virtual_node_resolves_to_its_baseline_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "baseline_shard"
            shard.mkdir()
            common = {
                "alpha": 0.8, "theta": 1.0, "aspect_ratio": 2.0,
                "source_runs": [str(shard)], "qa": {"precision_pass": True},
                "estimator_contract": NODE_ESTIMATE_CONTRACT,
                "cell_features": {}, "incoming_law": {}, "incoming_law_energy": {},
            }
            baseline = dict(common, ensemble_id=0)
            virtual = dict(
                common, ensemble_id=1,
                excitation={"family": "A_cu", "eta": 0.4})
            (root / "alpha_base.json").write_text(json.dumps(baseline))
            (root / "alpha_virtual.json").write_text(json.dumps(virtual))
            loaded = _load_node_estimates(
                root, {(0.8, 1.0, 2.0, 0): [shard]})
            self.assertEqual([row["ensemble_id"] for row in loaded], [0, 1])

    def test_virtual_node_uses_sentinel_not_pointwise_precision_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "baseline_shard"
            shard.mkdir()
            common = {
                "alpha": 0.8, "theta": 1.0, "aspect_ratio": 2.0,
                "source_runs": [str(shard)],
                "estimator_contract": NODE_ESTIMATE_CONTRACT,
                "cell_features": {}, "incoming_law": {}, "incoming_law_energy": {},
            }
            baseline = dict(common, ensemble_id=0, qa={"precision_pass": True})
            virtual = dict(common, ensemble_id=1,
                           excitation={"family": "A_cu", "eta": 0.25,
                                       "usable": True},
                           excitation_status="pass",
                           qa={"precision_pass": False, "sentinel_pass": True})
            (root / "alpha_base.json").write_text(json.dumps(baseline))
            (root / "alpha_virtual.json").write_text(json.dumps(virtual))
            loaded = _load_node_estimates(root, {(0.8, 1.0, 2.0, 0): [shard]})
            self.assertEqual(len(loaded), 2)

    def test_only_heldout_model_form_failure_is_admissible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "baseline_shard"; shard.mkdir()
            common = {
                "alpha": 1.0, "theta": 0.2, "aspect_ratio": 2.0,
                "source_runs": [str(shard)],
                "estimator_contract": NODE_ESTIMATE_CONTRACT,
                "cell_features": {}, "incoming_law": {}, "incoming_law_energy": {},
            }
            baseline = dict(common, ensemble_id=0, qa={"precision_pass": True})
            physical = ("angular_projection_pass", "elastic_pass",
                        "energy_projection_pass", "ess_pass",
                        "incoming_partition_pass", "memory_diagnostic_pass",
                        "propensity_pass", "proposal_balance_pass")
            qa = {name: True for name in physical}
            qa.update(sentinel_pass=False, precision_pass=False,
                      continuation_reasons=["model_form"])
            virtual = dict(common, ensemble_id=1, excitation_status="pass",
                           excitation={"family": "a2_tr", "eta": -0.5,
                                       "usable": True}, qa=qa)
            (root / "alpha_base.json").write_text(json.dumps(baseline))
            (root / "alpha_virtual.json").write_text(json.dumps(virtual))
            groups = {(1.0, 0.2, 2.0, 0): [shard]}
            self.assertEqual(len(_load_node_estimates(root, groups)), 2)
            virtual["qa"]["continuation_reasons"] = ["ess"]
            virtual["qa"]["ess_pass"] = False
            (root / "alpha_virtual.json").write_text(json.dumps(virtual))
            with self.assertRaisesRegex(ValueError, "has not passed closure QA"):
                _load_node_estimates(root, groups)

    def test_old_schema_22_estimate_is_not_mistaken_for_current_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shard"
            shard.mkdir()
            estimate = {
                "alpha": 0.8, "theta": 1.0, "aspect_ratio": 2.0,
                "source_runs": [str(shard)], "schema_version": "2.2.0",
                "qa": {"precision_pass": True},
            }
            (root / "alpha_old.json").write_text(json.dumps(estimate))
            with self.assertRaisesRegex(ValueError, "estimator contract"):
                _load_node_estimates(root, {(0.8, 1.0, 2.0): [shard]})


if __name__ == "__main__":
    unittest.main()
