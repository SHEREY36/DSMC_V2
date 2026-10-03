"""Encounter-unit clock and angular-memory runtime pieces."""
import json

import numpy as np
import pytest

from dsmc_v2.angular_memory import AngularMemoryTable, PARAMETER_NAMES, sample_cosine
from dsmc_v2.encounter import EncounterClock

TABLE = "DSMC_0D_v2/models/encounter_cross_section_v2.json"
TABLE_V1 = "DSMC_0D_v2/models/encounter_cross_section_v1.json"


def test_clock_reproduces_measured_nodes():
    clock = EncounterClock(TABLE)
    data = json.load(open(TABLE))
    for key, row in data["cross_section"].items():
        for theta, sigma in zip(row["theta"], row["sigma_enc_over_pi_d2"]):
            assert clock.sigma(theta, float(key), 1.0) / np.pi == pytest.approx(sigma, rel=1e-5)
    # geometry scales as d^2
    assert clock.sigma(1.0, 2.0, 0.5) == pytest.approx(0.25 * clock.sigma(1.0, 2.0, 1.0))
    # rotating rods sweep more area when rotation is relatively hot (low theta)
    assert clock.sigma(0.2, 3.0, 1.0) > clock.sigma(1.0, 3.0, 1.0) > clock.sigma(2.0, 3.0, 1.0)
    mid = clock.sigma(1.0, 2.25, 1.0)
    assert clock.sigma(1.0, 2.0, 1.0) < mid < clock.sigma(1.0, 2.5, 1.0)
    with pytest.raises(ValueError):
        clock.sigma(1.0, 3.5, 1.0)


def test_measured_table_agrees_with_campaign_table():
    """v2 (geometry x dynamic factor) and v1 (polynomial x ratio) agree at nodes."""
    new, old = EncounterClock(TABLE), EncounterClock(TABLE_V1)
    data = json.load(open(TABLE))
    for key, row in data["cross_section"].items():
        for theta in row["theta"]:
            assert new.sigma(theta, float(key), 1.0) == pytest.approx(
                old.sigma(theta, float(key), 1.0), rel=5e-3)


def _moments(a, b):
    x, w = np.polynomial.legendre.leggauss(200)
    p2 = 0.5 * (3 * x * x - 1)
    f = np.exp(a * x + b * p2) * w
    return float(f @ x / f.sum()), float(f @ p2 / f.sum())


@pytest.mark.parametrize("a,b", [(0.4, 0.3), (-0.8, 1.2), (1.5, -0.7)])
def test_cosine_sampler_matches_exponential_family(a, b):
    rng = np.random.default_rng(3)
    # z_in = z_out = 0 leaves the intercepts; wide support gate
    stencil = [(1.0, np.array([a, 0.0, 0.0, b, 0.0, 0.0, -50, 50, -50, 50]))]
    draws = np.array([sample_cosine(stencil, 0.0, 0.0, rng) for _ in range(40000)])
    mc, mp = _moments(a, b)
    assert np.mean(draws) == pytest.approx(mc, abs=0.012)
    assert np.mean(0.5 * (3 * draws**2 - 1)) == pytest.approx(mp, abs=0.012)


def test_support_gate_clamps_extrapolated_memory():
    rng = np.random.default_rng(5)
    # a huge rho1 would push a to +100 for z_in=1; the gate holds it at 0.5
    stencil = [(1.0, np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, -0.5, 0.5, -0.1, 0.1]))]
    draws = np.array([sample_cosine(stencil, 1.0, 0.0, rng) for _ in range(20000)])
    assert np.mean(draws) == pytest.approx(_moments(0.5, 0.0)[0], abs=0.015)


def test_table_binding_fails_closed(tmp_path):
    node = {"alpha": 0.8, "theta": 1.0, "AR": 2.0, "a_support": [-1, 1], "b_support": [-1, 1],
            **{name: 0.0 for name in PARAMETER_NAMES}}
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"schema": "angular-memory-v1",
                                "parameter_names": list(PARAMETER_NAMES), "nodes": [node]}))
    table = AngularMemoryTable(path)
    table.bind(np.array([[0.8, 1.0, 2.0]]))
    with pytest.raises(ValueError):
        table.bind(np.array([[0.8, 1.0, 2.0], [0.8, 2.0, 2.0]]))
