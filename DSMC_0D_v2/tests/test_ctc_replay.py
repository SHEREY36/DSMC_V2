import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsmc_v2.ctc_replay import CollisionFluxReservoir, REPLAY_WIDTH


class CollisionFluxReplayTests(unittest.TestCase):
    def test_fixed_reservoir_writes_normalised_verified_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            sampler = CollisionFluxReservoir(
                Path(temporary) / "state", [(1.0, 2.0)], 100, 17,
                alpha=0.8, aspect_ratio=2.0, mass=1.0, inertia=1.0)
            for index in range(150):
                sign = -1.0 if index % 2 else 1.0
                sampler.observe(
                    tau=1.5, v1=np.array([sign, 0.2, 0.0]),
                    v2=np.array([-sign, -0.2, 0.0]),
                    omega1=np.array([0.0, 1.0, 0.0]),
                    omega2=np.array([0.0, -1.0, 0.0]),
                    axis1=np.array([1.0, 0.0, 0.0]),
                    axis2=np.array([1.0, 0.0, 0.0]),
                    mean_velocity=np.zeros(3), trot=0.5)
            output = sampler.finalize()[0]
            binary = Path(output["binary"])
            metadata = json.loads(Path(output["metadata"]).read_text())
            self.assertEqual(binary.stat().st_size, 100 * REPLAY_WIDTH * 8)
            self.assertEqual(metadata["source_pairs_seen"], 150)
            self.assertEqual(metadata["record_count"], 100)
            self.assertAlmostEqual(metadata["temperature_rotational"], 1.0, places=12)
            self.assertTrue(metadata["qa"]["pass"])
            self.assertEqual(
                metadata["temperature_estimator"],
                "self_normalized_inverse_speed_v1")
            self.assertEqual(
                metadata["cell_measure_debiasing"],
                "inverse_physical_selection_speed_v1")

    def test_cell_measure_uses_pre_normalisation_selection_speed(self):
        with tempfile.TemporaryDirectory() as temporary:
            sampler = CollisionFluxReservoir(
                Path(temporary) / "state", [(1.0, 2.0)], 100, 17,
                alpha=0.8, aspect_ratio=2.0, mass=1.0, inertia=1.0)
            # The source populations have normalized relative speeds 1 and 2,
            # but physical selection speeds 1 and 4. Correct debiasing gives
            # the slow population four times the cell weight, not two.
            for index in range(100):
                trot = 1.0 if index < 50 else 4.0
                speed = 1.0 if index < 50 else 4.0
                sampler.observe(
                    tau=1.5,
                    v1=np.array([0.5 * speed, 0.0, 0.0]),
                    v2=np.array([-0.5 * speed, 0.0, 0.0]),
                    omega1=np.array([0.0, np.sqrt(2.0 * trot), 0.0]),
                    omega2=np.array([0.0, -np.sqrt(2.0 * trot), 0.0]),
                    axis1=np.array([1.0, 0.0, 0.0]),
                    axis2=np.array([1.0, 0.0, 0.0]),
                    mean_velocity=np.zeros(3), trot=trot)
            output = sampler.finalize()[0]
            self.assertTrue(output["qa"]["pass"])
            self.assertAlmostEqual(
                output["temperature_translational"], 0.4 / 3.0,
                places=12)
            np.testing.assert_allclose(
                sampler.selection_speed[0][:50], 1.0)
            np.testing.assert_allclose(
                sampler.selection_speed[0][50:], 4.0)

    def test_overlapping_windows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                CollisionFluxReservoir(
                    Path(temporary) / "state", [(0.0, 2.0), (1.0, 3.0)],
                    100, 4, alpha=0.8, aspect_ratio=2.0,
                    mass=1.0, inertia=1.0)


if __name__ == "__main__":
    unittest.main()
