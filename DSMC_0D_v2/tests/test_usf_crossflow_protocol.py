import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "DSMC_0D_v2" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MANIFEST = load_script("make_usf_study_manifest")
ANALYZE = load_script("analyze_usf_study")


def test_frozen_usf_design_sizes_and_interpolation_coverage(tmp_path):
    reference = MANIFEST.load_initialization_reference(
        ROOT / "DSMC_0D_v2/reference/usf_dem_fresh_v1.csv")
    expected = {"numerics": 108, "pilot": 168, "full": 736}
    for mode, count in expected.items():
        rows = MANIFEST.campaign_rows(
            mode, tmp_path / mode, 10_000, 0.01,
            2 if mode == "numerics" else 4, reference)
        assert len(rows) == count
        assert [row["task_id"] for row in rows] == list(range(count))
        assert all(float(row["alpha"]) < 1.0 for row in rows)
    full = MANIFEST.campaign_rows(
        "full", tmp_path / "full2", 10_000, 0.01, 4, reference)
    holdouts = [row for row in full
                if row["coordinate_role"] == "interpolation_holdout"]
    assert len(holdouts) == 96
    assert {float(row["aspect_ratio"]) for row in holdouts} == {
        1.15, 1.275, 1.425, 1.75, 2.25, 2.75}


def test_usf_run_reduction_uses_kinetic_stress_and_energy_ledger(tmp_path):
    prefix = tmp_path / "case"
    tau = np.linspace(0.0, 10.0, 51)
    time = 2.0 * tau
    ttr = np.full_like(tau, 2.0)
    trot = np.full_like(tau, 1.0)
    total = (3.0 * ttr + 2.0 * trot) / 5.0
    np.savetxt(Path(str(prefix) + ".txt"),
               np.column_stack((time, tau, ttr, trot, total)))
    kinetic = np.tile([1.6, -0.5, 0.0, 0.7, 0.0, 0.7], (len(tau), 1))
    # Deliberately huge proxy collisional stress: it must not contaminate Pk.
    collisional = np.full_like(kinetic, 100.0)
    np.savetxt(Path(str(prefix) + "_pressure.txt"),
               np.column_stack((time, tau, kinetic, collisional)))
    q = np.tile([0.1, 0.0, 0.0, -0.05, 0.0, -0.05], (len(tau), 1))
    np.savetxt(Path(str(prefix) + "_orientation.txt"),
               np.column_stack((time, tau, q)))
    np.savetxt(Path(str(prefix) + "_energy.txt"), np.column_stack((
        time, tau, np.full_like(tau, 100.0), tau, -tau, np.zeros_like(tau))))
    Path(str(prefix) + ".json").write_text(json.dumps({
        "number_density": 0.5,
        "particles": 1000,
        "termination_reason": "collision_target",
        "negative_energy_repairs": 0,
        "energy_axis_clamps": 0,
        "energy_monotonic_repairs": 0,
        "correction_fallback_fraction_in_evaluation_window": 0.0,
        "usf_energy_ledger": {"relative_residual": 0.0},
        "ntc": {"repeated_particle_pair_fraction": 0.0},
        "artifact_sha256": "artifact",
        "reference_provenance_sha256": "reference",
        "validation_case": {"particle_mass": 1.0, "particle_diameter": 1.0},
    }))
    row = {
        "task_id": "0", "mode": "pilot", "coordinate_role": "test",
        "arm": "corrected", "sphere": "false", "aspect_ratio": "2.0",
        "alpha": "0.8", "dt": "0.005", "rate_scale": "1.0",
        "initial_branch": "cold", "replicate": "0", "particles": "1000",
        "shear_rate": "0.5", "evaluation_start_tau": "6.0",
        "output_prefix": str(prefix),
    }
    record = ANALYZE.analyze_run(row)
    assert np.isclose(record["means"]["Tstar"], 8.0)
    assert np.isclose(record["means"]["Pk_xx"], 1.6)
    assert np.isclose(record["means"]["Pk_xy"], -0.5)
    assert np.isclose(record["means"]["theta"], 2.0)
    assert record["energy_ledger_increment_residual"] < 1e-14
    assert record["steady_energy_power_imbalance"] < 1e-14
    assert record["physical_run_pass"]
