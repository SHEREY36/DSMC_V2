"""The angular-memory I-projection recovers known natural parameters."""
import numpy as np

from coll_models_v2.angular_memory_fit import fit, moments


def test_recovers_memory_parameters():
    rng = np.random.default_rng(11)
    n = 60000
    z_in, z_out = rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)
    truth = np.array([0.2, 1.0, -0.6, -0.3, 0.8, 0.5])   # eta1 xi rho1 eta2 zeta rho2
    a = truth[0] + truth[1] * z_out + truth[2] * z_in
    b = truth[3] + truth[4] * z_out + truth[5] * z_in
    grid = np.linspace(-1, 1, 2001)
    cosine = np.empty(n)
    for i in range(n):
        pdf = np.exp(a[i] * grid + b[i] * 0.5 * (3 * grid**2 - 1))
        cdf = np.cumsum(pdf); cdf /= cdf[-1]
        cosine[i] = np.interp(rng.random(), cdf, grid)
    model = fit(cosine, z_in, z_out)
    assert model["grad"] < 1e-9
    assert np.allclose(model["theta"], truth, atol=0.12)
    # moment matching: the fitted law reproduces the observed E[z P2]
    mc, mp, *_ = moments(model["theta"][0] + model["theta"][1] * z_out + model["theta"][2] * z_in,
                         model["theta"][3] + model["theta"][4] * z_out + model["theta"][5] * z_in)
    p2 = 0.5 * (3 * cosine**2 - 1)
    assert abs(np.mean(z_in * mp) - np.mean(z_in * p2)) < 1e-6
