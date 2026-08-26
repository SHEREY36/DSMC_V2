import unittest

import numpy as np

from dsmc_v2.ntc import acceptance_probability, sample_distinct_pair


class NTCTests(unittest.TestCase):
    def test_distinct_unordered_pairs_are_uniform(self):
        rng = np.random.default_rng(4)
        counts = {(0, 1): 0, (0, 2): 0, (1, 2): 0}
        for _ in range(60000):
            counts[sample_distinct_pair(3, rng)] += 1
        self.assertLess(max(counts.values()) - min(counts.values()), 700)

    def test_majorant_violation_is_not_silently_clipped(self):
        with self.assertRaises(RuntimeError):
            acceptance_probability(2.0, 2.0, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()

