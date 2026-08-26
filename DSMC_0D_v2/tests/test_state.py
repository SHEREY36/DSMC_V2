import unittest

import numpy as np

from dsmc_v2.state import ParticleState


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


if __name__ == "__main__":
    unittest.main()
