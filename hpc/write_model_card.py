#!/usr/bin/env python3
"""Make a model folder self-describing: model_card.json, model.yaml, SHA256SUMS.

    hpc/python.sh hpc/write_model_card.py models/microscopic_closure_v2_final_v1

model_card.json lists the tables the artifact needs (the simulation refuses to
run the artifact without them), the grid, the fitting choices and provenance.
model.yaml is a ready configuration whose paths are relative to the folder, so
the folder can be copied anywhere and run with DSMC_0D_v2.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

FILES = ("closure_v2.npz", "encounter_cross_section.json", "angular_memory.json",
         "loss_memory.json", "manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    folder = Path(sys.argv[1])
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        raise SystemExit(f"{folder} lacks {missing}")
    manifest = json.loads((folder / "manifest.json").read_text())
    loss = json.loads((folder / "loss_memory.json").read_text())
    nodes = [(n["alpha"], n["theta"], n["AR"]) for n in loss["nodes"]]
    grid = {"alpha": sorted({n[0] for n in nodes}), "aspect_ratio": sorted({n[2] for n in nodes}),
            "theta_by_aspect_ratio": {str(ar): sorted({n[1] for n in nodes if n[2] == ar})
                                      for ar in sorted({n[2] for n in nodes})}}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    card = {
        "name": folder.name,
        "event_unit": "encounter",
        "requires": ["encounter_cross_section", "angular_memory", "loss_memory"],
        "n_nodes": len(nodes),
        "contact_model_id": loss.get("contact_model_id", "R1-legacy"),
        "grid": grid,
        "laws": {
            "clock": "sigma_c(theta, AR) = mean projected excluded area x measured dynamic factor "
                     "(encounter_cross_section.json)",
            "loss": ("eps ~ Beta(kappa mu, kappa (1 - mu)), mu = E[eps|z] = c_t z + c_r (1 - z), "
                     "kappa per node (loss_memory.json, schema loss-memory-v2)"
                     if loss.get("schema") == "loss-memory-v2" else
                     "eps = E[eps|z] B/<B>, E[eps|z] = c_t z + c_r (1 - z), B ~ Beta(1.21, 3.67) "
                     "(loss_memory.json; node means are plain per-encounter means)"),
            "exchange": "conditional I-projection p(z'|z, eps) per node, fitted with the "
                        "post-collision energy weight E_f (closure_v2.npz)",
            "angle": "p(c|z, z') ~ exp[(eta1 + xi z' + rho1 z) c + (eta2 + zeta z' + rho2 z) P2(c)], "
                     "energy-tilted (angular_memory.json)",
            "between_nodes": "physical stencil: Wasserstein average of energy quantiles, "
                             "mixture of angular laws, linear mixture of loss rates",
        },
        "note": "closure_v2.npz energy_mean_loss holds E_f-weighted node means of the loss; the "
                "runtime takes the loss scale from loss_memory.json, so the artifact must be run "
                "with that table (enforced through this card).",
        "provenance": {"git_commit": commit, "artifact_schema": manifest.get("schema_version"),
                       "stability_pass": manifest.get("stability_pass")},
        "sha256": {name: sha256(folder / name) for name in FILES},
    }
    (folder / "model_card.json").write_text(json.dumps(card, indent=1) + "\n")
    (folder / "SHA256SUMS").write_text("".join(f"{card['sha256'][n]}  {n}\n" for n in FILES))
    config = yaml.safe_load(Path("DSMC_0D_v2/config/encounter_unit_model_final.yaml").read_text())
    config["microscopic_closure"].update({
        "artifact": "closure_v2.npz", "encounter_cross_section": "encounter_cross_section.json",
        "angular_memory": "angular_memory.json", "loss_memory": "loss_memory.json"})
    (folder / "model.yaml").write_text(
        "# Paths are relative to this folder: run DSMC_0D_v2 from inside it, or prefix them.\n"
        + yaml.safe_dump(config, sort_keys=False))
    print(f"{folder}: model_card.json, model.yaml, SHA256SUMS written ({len(nodes)} nodes)")


if __name__ == "__main__":
    main()
