"""Loss law with incoming-partition memory."""
import json

import numpy as np
import pytest

from dsmc_v2.loss_memory import LossMemoryTable


def table(tmp_path, nodes):
    path = tmp_path / "loss.json"
    path.write_text(json.dumps({"schema": "loss-memory-v1", "nodes": nodes}))
    return LossMemoryTable(path)


def node(alpha, theta, ar, c_t, c_r, z_mean=0.5):
    mean = c_t * z_mean + c_r * (1 - z_mean)
    return {"alpha": alpha, "theta": theta, "AR": ar, "c_t": c_t, "c_r": c_r, "eps_mean": mean}


def test_scale_preserves_the_node_mean_and_follows_z(tmp_path):
    loss = table(tmp_path, [node(0.5, 1.0, 1.1, 0.37, 0.01)])
    loss.bind(np.array([[0.5, 1.0, 1.1]]))
    rates = loss.stencil({"energy_vertex_indices": [0], "energy_vertex_weights": [1.0]})
    z = np.linspace(0.0, 1.0, 1001)
    s = loss.scale(rates, z)
    assert s.mean() == pytest.approx(1.0, rel=1e-12)       # node mean unchanged for z ~ U(0,1), z_mean = 1/2
    assert s[0] < 0.1 and s[-1] > 1.9                        # a near-sphere loses almost only translation


def test_mixture_combines_conditional_means_not_ratios(tmp_path):
    loss = table(tmp_path, [node(0.95, 1.0, 2.0, 0.03, 0.02), node(1.0, 1.0, 2.0, 1e-7, 1e-7)])
    loss.bind(np.array([[0.95, 1.0, 2.0], [1.0, 1.0, 2.0]]))
    rates = loss.stencil({"energy_vertex_indices": [0, 1], "energy_vertex_weights": [0.5, 0.5]})
    # the elastic node carries no loss, so it must not dilute the shape of the dissipative one
    assert loss.scale(rates, 1.0) == pytest.approx(0.03 / 0.025)


def test_missing_node_fails_closed(tmp_path):
    loss = table(tmp_path, [node(0.5, 1.0, 2.0, 0.2, 0.2)])
    with pytest.raises(ValueError):
        loss.bind(np.array([[0.5, 1.0, 3.0]]))
