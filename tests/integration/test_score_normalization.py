import math
import unittest

import numpy as np

from dsmc_v2_contracts.features import TENSOR_BASIS


def states(rng, count):
    c = rng.normal(scale=math.sqrt(0.5), size=(count, 3))
    u = rng.normal(size=(count, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    omega = rng.normal(scale=math.sqrt(0.5), size=(count, 3))
    omega -= np.einsum("ni,ni->n", omega, u)[:, None] * u
    return c, omega, u


class ScoreNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c, cls.omega, cls.u = states(np.random.default_rng(4102), 250000)

    def test_five_spin_alignment_dual_jacobians(self):
        for basis in TENSOR_BASIS:
            cac = np.einsum("ni,ij,nj->n", self.c, basis, self.c)
            wow = np.einsum("ni,ij,nj->n", self.omega, basis, self.omega)
            uau = np.einsum("ni,ij,nj->n", self.u, basis, self.u)
            score_r = (10.0 * wow + 5.0 * uau) / 7.0
            score_q = (10.0 * wow + 40.0 * uau) / 7.0
            dpi = 2.0 * np.einsum("ni,nj,n->ij", self.c, self.c, cac) / len(cac)
            dr_r = 3.0 * np.einsum("ni,nj,n->ij", self.omega, self.omega, score_r) / len(cac)
            dq_r = 1.5 * np.einsum("ni,nj,n->ij", self.u, self.u, score_r) / len(cac)
            dr_q = 3.0 * np.einsum("ni,nj,n->ij", self.omega, self.omega, score_q) / len(cac)
            dq_q = 1.5 * np.einsum("ni,nj,n->ij", self.u, self.u, score_q) / len(cac)
            self.assertLess(np.linalg.norm(dpi - basis), 0.045)
            self.assertLess(np.linalg.norm(dr_r - basis), 0.05)
            self.assertLess(np.linalg.norm(dq_r), 0.05)
            self.assertLess(np.linalg.norm(dr_q), 0.08)
            self.assertLess(np.linalg.norm(dq_q - basis), 0.08)

    def test_three_heat_flux_directions(self):
        x = np.einsum("ni,ni->n", self.c, self.c)
        y = np.einsum("ni,ni->n", self.omega, self.omega)
        tr = self.c * (x[:, None] - 2.5)
        rot = self.c * (y[:, None] - 1.0)
        np.testing.assert_allclose(0.8 * tr.T @ tr / len(tr), np.eye(3), atol=0.045)
        np.testing.assert_allclose(2.0 * rot.T @ rot / len(rot), np.eye(3), atol=0.045)
        np.testing.assert_allclose(tr.T @ rot / len(tr), np.zeros((3, 3)), atol=0.03)


if __name__ == "__main__":
    unittest.main()

