import unittest

import numpy as np

from dsmc_v2.state import ParticleState, initialize_particles
from dsmc_v2_contracts import cell_features


class StateTests(unittest.TestCase):
    def test_axis_spin_and_scalar_energy_remain_consistent(self):
        state = ParticleState(
            np.zeros((2, 3)), np.array([1.0, 2.0]),
            np.array([[0.0, np.sqrt(2.0), 0.0], [0.0, 0.0, 2.0]]),
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 1.0)
        state.advance_axes(0.1)
        state.normalize_constraints()
        np.testing.assert_allclose(np.linalg.norm(state.axis, axis=1), 1.0)
        np.testing.assert_allclose(np.einsum("ni,ni->n", state.omega, state.axis), 0.0,
                                   atol=1.0e-13)

    def test_variational_initial_rotation_is_isotropic_and_tangent(self):
        np.random.seed(8128)
        state = initialize_particles(
            100000, 1.0, 1.0, 1.0, 0.4, np.random.default_rng(991),
            isotropic_rotation=True)
        state.normalize_constraints()
        features = cell_features(
            state.velocity, state.omega, state.axis, 1.0, 0.4, sphere=False)
        # QQ, RtRt and QRt exposed the old global-y/z representation: RtRt was
        # about 0.105 instead of zero even as N -> infinity.
        for index in (5, 6, 9):
            self.assertAlmostEqual(features[index], 0.0, delta=2.0e-3)


if __name__ == "__main__":
    unittest.main()
