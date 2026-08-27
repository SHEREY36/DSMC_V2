import json
import tempfile
import unittest
from pathlib import Path

from coll_models_v2.artifact import _load_node_estimates


class ArtifactInputTests(unittest.TestCase):
    def test_precomputed_node_must_cover_exact_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard1, shard2 = root / "pilot", root / "production"
            shard1.mkdir(); shard2.mkdir()
            estimate = {
                "alpha": 0.5, "theta": 0.1, "aspect_ratio": 1.1,
                "source_runs": [str(shard1), str(shard2)],
                "qa": {"precision_pass": True},
            }
            (root / "alpha_0.500_theta_0.100_AR_1.100.json").write_text(
                json.dumps(estimate))
            groups = {(0.5, 0.1, 1.1): [shard1, shard2]}
            self.assertEqual(len(_load_node_estimates(root, groups)), 1)
            groups[(0.5, 0.1, 1.1)].append(root / "continuation")
            with self.assertRaisesRegex(ValueError, "stale node estimate"):
                _load_node_estimates(root, groups)


if __name__ == "__main__":
    unittest.main()
