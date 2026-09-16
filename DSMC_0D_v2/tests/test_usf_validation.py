import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "DSMC_0D_v2" / "scripts" / "analyze_usf_validation.py"
SPEC = importlib.util.spec_from_file_location("analyze_usf_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_usf_output_columns_and_reduction(tmp_path):
    prefix = tmp_path / "case"
    tau = np.linspace(0.0, 10.0, 51)
    time = 2.0 * tau
    ttr = np.full_like(tau, 2.0)
    trot = np.full_like(tau, 1.0)
    total = (3.0 * ttr + 2.0 * trot) / 5.0
    np.savetxt(Path(str(prefix) + ".txt"),
               np.column_stack((time, tau, ttr, trot, total)))

    # n*Ttr = 1, so the stored dimensional kinetic pressure is also the
    # expected reduced pressure.  Collisional columns are zero here.
    kinetic = np.tile([1.6, -0.5, 0.0, 0.7, 0.0, 0.7], (len(tau), 1))
    collisional = np.zeros_like(kinetic)
    np.savetxt(Path(str(prefix) + "_pressure.txt"),
               np.column_stack((time, tau, kinetic, collisional)))
    q = np.tile([0.1, 0.0, 0.0, -0.05, 0.0, -0.05], (len(tau), 1))
    np.savetxt(Path(str(prefix) + "_orientation.txt"),
               np.column_stack((time, tau, q)))
    Path(str(prefix) + ".json").write_text(json.dumps({
        "number_density": 0.5,
        "negative_energy_repairs": 0,
        "energy_axis_clamps": 0,
        "energy_monotonic_repairs": 0,
        "out_of_domain_fraction": 0.0,
        "closure_overhead_fraction": 0.01,
        "runtime_seconds": 1.0,
        "artifact": "candidate.npz",
        "artifact_sha256": "abc123",
    }))
    row = {
        "task_id": "0", "mode": "pilot", "arm": "corrected",
        "alpha": "0.8", "aspect_ratio": "2.0", "replicate": "0",
        "seed": "1", "output_prefix": str(prefix),
    }
    record, series = MODULE.analyse_run(row)
    assert np.isclose(record["late_means"]["Pxx"], 1.6)
    assert np.isclose(record["late_means"]["Pyy"], 0.7)
    assert np.isclose(record["late_means"]["Pzz"], 0.7)
    assert np.isclose(record["late_means"]["Pxy"], -0.5)
    assert np.isclose(record["late_means"]["theta"], 2.0)
    assert np.isclose(record["late_means"]["nematic_order"], 0.1)
    assert record["finite"]
    assert record["artifact_sha256"] == "abc123"
    assert series["pressure"].shape == (51, 4)
