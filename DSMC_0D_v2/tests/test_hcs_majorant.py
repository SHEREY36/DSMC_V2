import yaml
from pathlib import Path

from dsmc_v2.particle import particle_parameters
from dsmc_v2.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[2]


def test_rescaled_hcs_keeps_ntc_majorant_stationary(tmp_path):
    """Rescaling undoes each step's cooling, so the majorant must not ratchet.

    Multiplying vrmax by the reheating factor grew it as exp(~0.08(1-a^2)tau):
    about 50x by tau=60 at alpha=0.5, and the candidate arrays with it, which
    OOM-killed every alpha<=0.9 task of the first HCS-NG sweep.
    """
    config = yaml.safe_load(
        (ROOT / "DSMC_0D_v2/config/full_domain_baseline_candidate.yaml").read_text())
    config["particle"]["AR"] = 1.0
    config["simulation"].update(sphere_collision=True, hcs_rescale_temperature=True)
    # Same routing the HCS-NG runner uses for its sphere arm.
    config["microscopic_closure"].update(routing="legacy_rank0", angular="legacy",
                                         invariant_corrections=False)
    config["system"].update(alpha=0.5, kTt=1.0, kTr=1.0, domain=[30.0, 30.0, 30.0])
    params = particle_parameters(config)
    config["system"]["phi"] = (600 - 0.25) * params.volume / 30.0**3
    config["time"].update(dt=0.02, dtau=1.0, t_end=1.0e6, tau_end=60.0)
    result = run_simulation(config, 3, tmp_path / "sphere.txt")
    ntc = result["ntc"]
    assert result["cpp"] >= 60.0
    assert ntc["final_vrmax_over_initial"] < 1.5, ntc
    assert ntc["majorant_violation_fraction"] < 1.0e-4, ntc
    assert result["peak_rss_mib"] > 0.0
