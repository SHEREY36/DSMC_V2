import copy
import unittest

import numpy as np

from dsmc_v2.kernel import SpherocylinderKernel
from dsmc_v2.particle import ParticleParameters
from dsmc_v2.state import ParticleState


class _GMM:
    def sample_conditionals(self, r, e_tr, e_r1, n_samples=1):
        return np.array([[0.55, 0.35]])


class _Models:
    cond_gmm = _GMM()

    @staticmethod
    def loss_parameters(alpha, aspect_ratio):
        return {"gamma_max": 0.4, "one_hit_probability": 0.8}


class _Closure:
    @staticmethod
    def alpha_eff(alpha, aspect_ratio):
        return 3.0

    @staticmethod
    def spin_directions(*args, **kwargs):
        return np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


def _state():
    velocity = np.array([[1.0, 0.2, 0.0], [-1.0, -0.2, 0.0]])
    energy = np.array([0.3, 0.4])
    omega = np.array([[0.0, np.sqrt(0.6), 0.0],
                      [0.0, np.sqrt(0.8), 0.0]])
    axis = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    return ParticleState(velocity, energy, omega, axis, 1.0)


class ConservativeKernelTests(unittest.TestCase):
    def setUp(self):
        self.params = ParticleParameters(2.0, 0.5, 1.0, 1.0, 1.0, 1.0,
                                         1.0, 5.0)
        self.normal = np.array([1.0, 0.0, 0.0])

    @staticmethod
    def _energy(state):
        vcm = np.mean(state.velocity, axis=0)
        return 0.5 * np.sum((state.velocity - vcm) ** 2) + np.sum(state.rotational_energy)

    def test_routing_changes_split_not_total_loss_draw(self):
        states = [_state(), _state()]
        kernels = [
            SpherocylinderKernel(self.params, _Models(), 0.8, 1.21, 3.67, 1.0,
                                 None, "legacy_rank0", "legacy",
                                 np.random.default_rng(9), np.random.default_rng(10)),
            SpherocylinderKernel(self.params, _Models(), 0.8, 1.21, 3.67, 1.0,
                                 None, "ctc_moment16", "legacy",
                                 np.random.default_rng(9), np.random.default_rng(10)),
        ]
        kernels[1].set_cell_routing(0.2)
        initial = self._energy(states[0])
        for state, kernel in zip(states, kernels):
            np.random.seed(44)
            v1, v2 = state.velocity.copy()
            relative = v1 - v2
            kernel.collide(state, 0, 1, self.normal, v1, v2, relative,
                           np.linalg.norm(relative), 1.0, 1.0)
        self.assertAlmostEqual(initial - self._energy(states[0]),
                               initial - self._energy(states[1]), places=13)
        self.assertNotAlmostEqual(np.sum(states[0].rotational_energy),
                                  np.sum(states[1].rotational_energy), places=8)

    def test_vss_changes_direction_not_translational_energy(self):
        states = [_state(), _state()]
        kernels = [
            SpherocylinderKernel(self.params, _Models(), 0.8, 1.21, 3.67, 1.0,
                                 None, "legacy_rank0", "legacy",
                                 np.random.default_rng(9), np.random.default_rng(10)),
            SpherocylinderKernel(self.params, _Models(), 0.8, 1.21, 3.67, 1.0,
                                 _Closure(), "legacy_rank0", "ctc_vss_rank2",
                                 np.random.default_rng(9), np.random.default_rng(10)),
        ]
        for state, kernel in zip(states, kernels):
            np.random.seed(55)
            v1, v2 = state.velocity.copy()
            relative = v1 - v2
            kernel.collide(state, 0, 1, self.normal, v1, v2, relative,
                           np.linalg.norm(relative), 1.0, 1.0)
        for state in states:
            self.assertAlmostEqual(np.dot(state.velocity[0] - state.velocity[1],
                                          state.velocity[0] - state.velocity[1]),
                                   np.dot(states[0].velocity[0] - states[0].velocity[1],
                                          states[0].velocity[0] - states[0].velocity[1]), places=13)
        self.assertFalse(np.allclose(states[0].velocity, states[1].velocity))


if __name__ == "__main__":
    unittest.main()
