import numpy as np
import yaml
from pathlib import Path

from dsmc_v2.particle import particle_parameters
from dsmc_v2.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[2]


def _config(ar, ktr, particles=800):
    config = yaml.safe_load(
        (ROOT / "DSMC_0D_v2/config/full_domain_baseline_candidate.yaml").read_text())
    config["particle"]["AR"] = ar
    config["system"].update(alpha=1.0, kTt=1.0, kTr=ktr, domain=[40.0, 40.0, 40.0])
    params = particle_parameters(config)
    config["system"]["phi"] = (particles - 0.25) * params.volume / 40.0**3
    config["preprocessing"]["model_root"] = str(ROOT / "DSMC_0D_v2/models")
    # The exact elastic block must not need the artifact at all.
    config["microscopic_closure"]["artifact"] = "/nonexistent/closure_v2.npz"
    config["time"].update(dt=0.01, dtau=1.0, t_end=1.0e5, tau_end=20.0)
    return config


def test_elastic_block_relaxes_to_equipartition_and_conserves_energy(tmp_path):
    for ar, ktr in ((1.1, 0.1), (3.0, 2.0)):
        path = tmp_path / f"elastic_{ar}.txt"
        result = run_simulation(_config(ar, ktr), 11, path)
        rows = np.loadtxt(path)
        assert result["routing"] == "elastic_bl"
        assert result["runtime_gate"]["pass"]
        assert np.isclose(
            result["ntc"]["initial_vrmax_temperature_bound"],
            1.0 + (2.0 / 3.0) * ktr)
        np.testing.assert_allclose(rows[:, 4], rows[0, 4], rtol=1.0e-9)
        late = rows[rows[:, 1] >= 10.0]
        theta = np.mean(late[:, 2] / late[:, 3])
        assert abs(theta - 1.0) < 0.05, theta


def test_elastic_limit_can_be_routed_back_to_closure():
    config = _config(2.0, 1.0)
    config["microscopic_closure"]["elastic_limit"] = "closure"
    try:
        run_simulation(config, 1, "/tmp/unused_elastic_closure.txt")
    except (FileNotFoundError, OSError):
        pass    # it tried to load the (missing) artifact, i.e. the closure path
    else:
        raise AssertionError("closure elastic limit did not use the artifact")
