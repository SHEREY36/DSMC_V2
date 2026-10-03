"""Parallel bootstrap refits give the replicate set of the serial loop."""
import numpy as np

from coll_models_v2 import estimate


def synthetic_events(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    z_in = rng.beta(2.0, 2.0, n)
    z_out = np.clip(0.6 * z_in + 0.4 * rng.beta(2.0, 2.0, n), 1e-6, 1 - 1e-6)
    return {"z_in": z_in, "z_el": z_in, "z_out": z_out, "loss": rng.uniform(0.05, 0.15, n),
            "energy": rng.gamma(4.0, 0.5, n), "cosine": rng.uniform(-1.0, 1.0, n),
            "weight": np.ones(n), "block": rng.integers(0, estimate.N_BLOCKS, n)}


def test_parallel_bootstrap_matches_serial():
    events = synthetic_events()
    serial = estimate._bootstrap(events, 20, 11, workers=1)
    parallel = estimate._bootstrap(events, 20, 11, workers=3)
    assert serial.keys() == parallel.keys()
    for name in serial:
        for field in ("standard_error", "ci_low", "ci_high"):
            assert np.isclose(serial[name][field], parallel[name][field], rtol=1e-10, atol=1e-12)


def test_exchange_fit_uses_post_collision_energy_weight():
    events = synthetic_events()
    assert estimate.EXCHANGE_WEIGHTING == "post_energy"
    weighted = estimate._fit(events, allow_joint=False, model_form=False)
    plain = estimate.EXCHANGE_WEIGHTING
    try:
        estimate.EXCHANGE_WEIGHTING = "none"
        unweighted = estimate._fit(events, allow_joint=False, model_form=False)
    finally:
        estimate.EXCHANGE_WEIGHTING = plain
    assert weighted["energy"]["lambda3"] != unweighted["energy"]["lambda3"]
