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
