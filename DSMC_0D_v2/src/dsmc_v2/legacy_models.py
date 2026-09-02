"""Load only the unchanged v1 conditional-GMM and BL loss inputs."""

from __future__ import annotations

from pathlib import Path

from coll_models_v2.legacy_bl import LegacyBL

from .gmm_energy import ConditionalGMM


SUPPORTED_ASPECT_RATIOS = {
    1.1: "AR11", 1.25: "AR125", 1.5: "AR15",
    2.0: "AR20", 2.5: "AR25", 3.0: "AR30",
}


class FrozenLossModel:
    """Load only the preserved scalar BL loss tables, never the legacy GMM."""

    def __init__(self, model_root: str | Path,
                 beta_a: float = 1.21, beta_b: float = 3.67):
        root = Path(model_root)
        self.bl = LegacyBL.load(root / "dissipation" / "gamma_max_table.json",
                                root / "dissipation" / "one_hit_table.json",
                                beta_a, beta_b)

    def loss_parameters(self, alpha: float, aspect_ratio: float) -> dict[str, float]:
        return self.bl.parameters(alpha, aspect_ratio)


class LegacyModels:
    def __init__(self, model_root: str | Path, aspect_ratio: float,
                 beta_a: float = 1.21, beta_b: float = 3.67):
        root = Path(model_root)
        matches = [tag for ar, tag in SUPPORTED_ASPECT_RATIOS.items()
                   if abs(ar - float(aspect_ratio)) <= 1.0e-12]
        if not matches:
            raise ValueError(f"no unchanged v1 GMM exists for AR={aspect_ratio}")
        self.cond_gmm = ConditionalGMM(root / "exchange_gmm" / f"gmm_cond_{matches[0]}.npz")
        self.bl = LegacyBL.load(root / "dissipation" / "gamma_max_table.json",
                                root / "dissipation" / "one_hit_table.json",
                                beta_a, beta_b)

    def loss_parameters(self, alpha: float, aspect_ratio: float) -> dict[str, float]:
        return self.bl.parameters(alpha, aspect_ratio)
