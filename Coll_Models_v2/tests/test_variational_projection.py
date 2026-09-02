import unittest

import numpy as np

from coll_models_v2.fit_exchange import fit_exchange_kernel
from coll_models_v2.artifact import _incoming_partition_mean
from coll_models_v2.projections import (
    angular_quantiles,
    energy_quantiles,
    fit_angular_projection,
    fit_energy_projection,
)


class VariationalProjectionTests(unittest.TestCase):
    def test_beta22_equilibrium_has_zero_natural_parameters(self):
        fit = fit_energy_projection(0.5, 0.3)
        np.testing.assert_allclose(fit.parameters, 0.0, atol=1.0e-10)

    def test_energy_projection_reproduces_ordinary_moments(self):
        fit = fit_energy_projection(0.4, 0.22)
        np.testing.assert_allclose(fit.moments, [0.4, 0.22], atol=1.0e-8)

    def test_energy_projection_rejects_infeasible_moments(self):
        with self.assertRaises(ValueError):
            fit_energy_projection(0.4, 0.1)

    def test_isotropic_angle_has_zero_natural_parameters(self):
        fit = fit_angular_projection(0.0, 0.0)
        np.testing.assert_allclose(fit.parameters, 0.0, atol=1.0e-10)

    def test_angular_projection_reproduces_two_moments(self):
        fit = fit_angular_projection(0.2, -0.1)
        np.testing.assert_allclose(fit.moments, [0.2, -0.1], atol=1.0e-7)

    def test_energy_quantile_sampler_reproduces_projection(self):
        fit = fit_energy_projection(0.63, 0.43)
        rng = np.random.default_rng(81)
        probability = np.linspace(0.0, 1.0, 4097)
        table = energy_quantiles(fit.parameters, probability)
        sample = np.interp(rng.random(250000), probability, table)
        self.assertAlmostEqual(np.mean(sample), 0.63, delta=0.002)
        self.assertAlmostEqual(np.mean(sample * sample), 0.43, delta=0.002)

    def test_angular_quantile_sampler_reproduces_projection(self):
        fit = fit_angular_projection(-0.21, 0.08)
        rng = np.random.default_rng(82)
        probability = np.linspace(0.0, 1.0, 4097)
        table = angular_quantiles(fit.parameters, probability)
        sample = np.interp(rng.random(250000), probability, table)
        self.assertAlmostEqual(np.mean(sample), -0.21, delta=0.002)
        self.assertAlmostEqual(np.mean(0.5 * (3.0 * sample**2 - 1.0)), 0.08, delta=0.002)

    def test_exchange_fit_uses_direct_probability_without_factor_two(self):
        rng = np.random.default_rng(83)
        n = 300000
        zin = rng.beta(2.0, 2.0, n)
        opened = rng.random(n) < 0.37
        zout = zin.copy()
        zout[opened] = rng.beta(3.0, 4.0, np.count_nonzero(opened))
        fit = fit_exchange_kernel(zin, zout, np.ones(n))
        self.assertAlmostEqual(fit["p_exch"], 0.37, delta=0.008)
        self.assertAlmostEqual(fit["reset_mean"], 3.0 / 7.0, delta=0.006)

    def test_exchange_fit_does_not_clip_invalid_rate(self):
        zin = np.linspace(0.1, 0.9, 100)
        zout = 1.1 * zin - 0.05
        with self.assertRaisesRegex(ValueError, "exchange probability"):
            fit_exchange_kernel(zin, zout, np.ones_like(zin))

    def test_unequal_scale_gamma_ratio_is_not_ratio_of_means(self):
        self.assertAlmostEqual(_incoming_partition_mean(1.0), 0.5, places=12)
        self.assertGreater(abs(_incoming_partition_mean(0.2) - 0.2 / 1.2), 0.04)


if __name__ == "__main__":
    unittest.main()
