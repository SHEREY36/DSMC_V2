"""Encounter loss with incoming-partition memory.

For one CTC encounter let z = E_tr/E be the incoming translational share and
eps the fraction of the pair energy lost.  At every artifact node the loss has
the conditional mean

    E[eps | z] = c_t z + c_r (1 - z),

the loss rates of translational and rotational energy, fitted on the CTC
encounters (Coll_Models_v2/scripts/fit_loss_memory_table.py).  Near the sphere
almost all of the loss comes out of translation (eps nearly proportional to z);
long rods lose from both modes almost equally.

The runtime keeps the loss draw on the encounter scale and multiplies it by

    s(z) = sum_k w_k (c_t,k z + c_r,k (1 - z)) / sum_k w_k eps_k,

the stencil mixture of the node conditional means over the mixture mean.  The
node mean loss is therefore unchanged, and the loss follows the pair's own
energy split.  Without it the loss is independent of z, which is exact only
at theta = 1, where z and E are independent; elsewhere the dissipation
<eps E> is off by the covariance of eps and E (2 % at AR 1.5 in USF).

Schema loss-memory-v2 (contact model C1) carries the loss law itself: at each
node a Beta concentration kappa, and the encounter loss is drawn as

    eps ~ Beta(kappa mu, kappa (1 - mu)),   mu = sum_k w_k (c_t,k z + c_r,k (1 - z)),

with kappa the stencil mixture of the node values.  The mean is mu exactly and
0 < eps < 1 by construction.  v1 tables keep the scaled-Beta draw above.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LossMemoryTable:
    """Per-node loss rates, combined on the closure's physical stencil."""

    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text())
        if data.get("schema") not in ("loss-memory-v1", "loss-memory-v2"):
            raise ValueError(f"{path} is not a loss-memory table")
        # v2: the loss is drawn from Beta(kappa mu, kappa (1 - mu)) (bounded law)
        self.bounded = data["schema"] == "loss-memory-v2"
        self.by_node = {}
        for node in data["nodes"]:
            kappa = node.get("kappa") if self.bounded else None
            values = np.array([node["c_t"], node["c_r"], node["eps_mean"],
                               np.nan if kappa is None else kappa], dtype=float)
            if not np.all(np.isfinite(values[:3])):
                raise ValueError(f"non-finite loss-memory rates at {node}")
            if self.bounded and float(node["alpha"]) < 1.0 and not np.isfinite(values[3]):
                raise ValueError(f"loss-memory-v2 node without kappa: {node}")
            self.by_node[self._key(node["alpha"], node["theta"], node["AR"])] = values
        self.path = str(path)
        self._node_rates = None

    @staticmethod
    def _key(alpha, theta, aspect_ratio):
        return (round(float(alpha), 4), round(float(theta), 5), round(float(aspect_ratio), 4))

    def bind(self, coordinates: np.ndarray) -> None:
        """Align with the artifact node order; fail closed on any gap."""
        rows = []
        for alpha, theta, aspect_ratio in np.asarray(coordinates, dtype=float):
            key = self._key(alpha, theta, aspect_ratio)
            if key not in self.by_node:
                raise ValueError(f"loss-memory table has no node {key}")
            rows.append(self.by_node[key])
        self._node_rates = np.asarray(rows)

    def stencil(self, state: dict) -> np.ndarray:
        """Mixture rates (c_t, c_r, eps_mean, kappa) on the energy kernel's stencil.
        kappa is nan for v1 tables; elastic vertices (no kappa) do not enter its mixture."""
        if self._node_rates is None:
            raise RuntimeError("loss-memory table is not bound to the artifact")
        indices = np.asarray(state["energy_vertex_indices"], dtype=int)
        weights = np.asarray(state["energy_vertex_weights"], dtype=float)
        weights = weights / float(np.sum(weights))
        rows = self._node_rates[indices]
        out = weights @ np.nan_to_num(rows, nan=0.0)
        has = np.isfinite(rows[:, 3])
        out[3] = float(weights[has] @ rows[has, 3] / weights[has].sum()) if has.any() else np.nan
        return out

    @staticmethod
    def draw(rates: np.ndarray, z_in: float, rng) -> float:
        """Bounded loss eps ~ Beta(kappa mu, kappa (1 - mu)), mu = c_t z + c_r (1 - z)."""
        c_t, c_r, _, kappa = (float(v) for v in rates)
        mu = c_t * z_in + c_r * (1.0 - z_in)
        if not mu > 1.0e-12 or not np.isfinite(kappa):
            return 0.0
        mu = min(mu, 1.0 - 1.0e-8)
        return float(rng.beta(kappa * mu, kappa * (1.0 - mu)))

    @staticmethod
    def scale(rates: np.ndarray, z_in) -> np.ndarray | float:
        """s(z) = E[eps | z] / E[eps]; 1 where the stencil carries no loss."""
        c_t, c_r, mean = (float(v) for v in rates[:3])
        if not mean > 1.0e-6:
            return np.ones_like(z_in) if np.ndim(z_in) else 1.0
        return np.maximum(c_t * np.asarray(z_in) + c_r * (1.0 - np.asarray(z_in)), 0.0) / mean
