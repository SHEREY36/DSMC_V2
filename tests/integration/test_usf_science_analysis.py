from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (Path(__file__).resolve().parents[2]
          / "DSMC_0D_v2" / "scripts" / "analyze_usf_science.py")
SPEC = importlib.util.spec_from_file_location("analyze_usf_science", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _orientation_row(q: np.ndarray) -> np.ndarray:
    return np.array([0.0, 0.0, q[0, 0], q[0, 1], q[0, 2],
                     q[1, 1], q[1, 2], q[2, 2]])


def test_nematic_order_averages_tensor_before_diagonalizing() -> None:
    axes = np.eye(3)
    values = np.asarray([
        _orientation_row(np.outer(axis, axis) - np.eye(3) / 3.0)
        for axis in axes])
    q = MODULE.mean_orientation_tensor(values)
    components = np.array([q[0, 0], q[0, 1], q[0, 2],
                           q[1, 1], q[1, 2], q[2, 2]])
    np.testing.assert_allclose(MODULE.nematic_order(components), 0.0,
                               atol=1.0e-15)


def test_nematic_order_is_one_for_perfect_alignment() -> None:
    q = np.diag([2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])
    components = np.array([q[0, 0], q[0, 1], q[0, 2],
                           q[1, 1], q[1, 2], q[2, 2]])
    np.testing.assert_allclose(MODULE.nematic_order(components), 1.0)


def test_branch_pairing_precedes_seed_standard_error() -> None:
    items = [
        {"replicate": 0, "branch": "cold", "value": 1.0},
        {"replicate": 0, "branch": "hot", "value": 3.0},
        {"replicate": 1, "branch": "cold", "value": 5.0},
        {"replicate": 1, "branch": "hot", "value": 7.0},
    ]
    mean, standard_error, branch_gap = MODULE.paired_estimate(items, "value")
    np.testing.assert_allclose(mean, 4.0)
    np.testing.assert_allclose(standard_error, 2.0)
    np.testing.assert_allclose(branch_gap, 0.5)
