import csv

import numpy as np

from dsmc_v2.non_gaussian import NonGaussianDiagnostics, reduced_observables
from dsmc_v2.state import ParticleState


def isotropic_state(seed=17, count=120000):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=(count, 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    velocity = rng.normal(size=(count, 3))
    omega = rng.normal(size=(count, 3))
    omega -= np.einsum("ni,ni->n", omega, axis)[:, None] * axis
    energy = 0.5 * np.einsum("ni,ni->n", omega, omega)
    return ParticleState(velocity, energy, omega, axis, 1.0)


def test_maxwellian_reduced_cumulants_are_zero():
    state = isotropic_state()
    values = reduced_observables(
        state.velocity, state.omega, state.axis, 1.0, 1.0)
    assert abs(values["a20"]) < 0.015
    assert abs(values["a02"]) < 0.015
    assert abs(values["a11"]) < 0.015
    assert abs(values["A_cu"]) < 0.015
    assert abs(values["A_cw_quadrupolar"]) < 0.015
    assert np.isclose(values["c2"], 1.5, atol=1e-12)
    assert np.isclose(values["w2"], 1.0, atol=1e-12)


def test_similarity_rescale_preserves_all_reduced_observables():
    state = isotropic_state(count=2000)
    before = reduced_observables(
        state.velocity, state.omega, state.axis, 1.0, 1.0)
    state.rescale_thermal_state(7.25)
    state.normalize_constraints()
    after = reduced_observables(
        state.velocity, state.omega, state.axis, 1.0, 1.0)
    for name in ("theta", "c2", "c4", "c6", "w2", "w4", "w6",
                 "c2w2", "a20", "a02", "a11", "A_cu",
                 "A_cw_quadrupolar"):
        assert np.isclose(before[name], after[name], rtol=2e-13, atol=2e-13)


def test_streaming_outputs_keep_counts_and_declared_window(tmp_path):
    state = isotropic_state(count=1000)
    config = {"diagnostics": {"non_gaussian": {
        "enabled": True, "sample_start_tau": 5.0,
        "sample_end_tau": 15.0, "sample_delta_tau": 5.0,
        "minimum_tail_count": 1,
    }}}
    output = tmp_path / "trajectory.txt"
    diagnostic = NonGaussianDiagnostics(config, output, state.count, 1.0, 1.0)
    assert not diagnostic.maybe_sample(0.0, 4.9, state)
    for tau in (5.0, 10.0, 15.0):
        assert diagnostic.maybe_sample(tau, tau, state)
    result = diagnostic.close()
    assert result["sampling_complete"]
    assert result["n_samples"] == 3
    assert result["n_particle_samples"] == 3000
    with (tmp_path / "trajectory_ng_moments.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with np.load(tmp_path / "trajectory_ng_histograms.npz") as data:
        assert data["c_counts"].sum() + int(data["c_overflow"]) == 3000
        assert data["w_counts"].sum() + int(data["w_underflow"]) \
            + int(data["w_overflow"]) == 3000


def test_invalid_similarity_scale_fails():
    state = isotropic_state(count=10)
    for value in (0.0, -1.0, np.nan):
        try:
            state.rescale_thermal_state(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid scaling was accepted")
