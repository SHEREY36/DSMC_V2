from pathlib import Path

import numpy as np
import yaml

from dsmc_v2.particle import particle_parameters
from dsmc_v2.simulation import run_simulation


ROOT = Path(__file__).resolve().parents[2]


def test_sphere_usf_writes_closed_energy_ledger(tmp_path):
    config = yaml.safe_load((ROOT / "DSMC_0D_v2/config/default.yaml").read_text())
    config["particle"]["AR"] = 1.0
    config["system"].update({
        "alpha": 0.8, "kTt": 1.0, "kTr": 0.0,
        "phi": 0.01, "particle_count": 128})
    config["simulation"].update({
        "sphere_collision": True, "exact_initial_temperatures": True})
    parameters = particle_parameters(config)
    box = (128 * parameters.volume / 0.01) ** (1.0 / 3.0)
    config["system"]["domain"] = [box, box, box]
    config["time"].update({
        "dt": 0.01, "dtau": 0.25, "t_end": 1.0e6, "tau_end": 2.0})
    config["flow"] = {"mode": "usf", "shear_rate": 0.05}
    # A sphere is an exact control and must not query this rod closure route.
    config["microscopic_closure"].update({
        "routing": "variational_v2", "angular": "variational_v2"})

    path = tmp_path / "sphere.txt"
    diagnostics = run_simulation(config, 81, path)
    energy = np.atleast_2d(np.loadtxt(tmp_path / "sphere_energy.txt"))
    assert energy.shape[1] == 6
    assert len(energy) >= 3
    assert diagnostics["particles"] == 128
    assert diagnostics["termination_reason"] == "collision_target"
    assert diagnostics["routing"] == "sphere_exact"
    assert diagnostics["collisional_stress_status"] == "exact_sphere_impulse_normal_v1"
    assert diagnostics["usf_energy_ledger"]["relative_residual"] < 1e-12
    np.testing.assert_allclose(
        energy[:, 2] - energy[0, 2],
        (energy[:, 3] - energy[0, 3]) + (energy[:, 4] - energy[0, 4]),
        rtol=2e-8, atol=2e-8)
