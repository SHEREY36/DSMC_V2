import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "make_manifest", Path(__file__).resolve().parents[2] / "hpc" / "make_manifest.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ManifestTests(unittest.TestCase):
    def test_grid_and_common_random_numbers(self):
        rows = MODULE.base_rows("pilot", 20000, 0, "results/ctc")
        self.assertEqual(len(rows), 870)
        line = [row for row in rows if row["theta"] == 0.5 and row["aspect_ratio"] == 2.0
                and row["role"] == "routing"]
        self.assertEqual(len({row["seed"] for row in line}), 1)
        self.assertEqual(len(line), 12)
        elastic = [row for row in rows if row["role"] == "vss_elastic_reference"]
        self.assertEqual(len(elastic), 6)


if __name__ == "__main__":
    unittest.main()
