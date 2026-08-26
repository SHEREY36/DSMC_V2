#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from dsmc_v2.artifact import CollisionArtifactV2
from dsmc_v2.simulation import HomogeneousDSMC
from dsmc_v2.state import ParticleState


def initial_state(count, ttr, trot, mass, moi, rng):
    velocity = rng.normal(scale=np.sqrt(ttr / mass), size=(count, 3))
    velocity -= np.mean(velocity, axis=0)
    axis = rng.normal(size=(count, 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    omega = rng.normal(size=(count, 3))
    omega -= np.einsum("ni,ni->n", omega, axis)[:, None] * axis
    current = moi * np.mean(np.einsum("ni,ni->n", omega, omega)) / 2.0
    omega *= np.sqrt(trot / current)
    return ParticleState(velocity, omega, axis)


def main():
    parser = argparse.ArgumentParser(description="Run DSMC_0D_v2")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    p, s, flow, time, collision, sim = [config[key] for key in
                                        ("particle", "system", "flow", "time", "collision", "simulation")]
    rng = np.random.default_rng(sim["seed"])
    state = initial_state(p["count"], s["translational_temperature"],
                          s["rotational_temperature"], p["mass"], p["moi_perpendicular"], rng)
    artifact = CollisionArtifactV2(collision["artifact"])
    engine = HomogeneousDSMC(state, artifact, s["alpha"], p["aspect_ratio"], p["mass"],
                             p["moi_perpendicular"], collision["mode"], sim["seed"])
    output = Path(sim["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time", "Ttr", "Trot", "theta", "candidates", "accepted",
                         "infeasible_draws", "max_energy_error", "max_angular_error"])
        for step in range(time["steps"] + 1):
            ttr, trot = state.temperatures(p["mass"], p["moi_perpendicular"])
            if step == 0:
                diagnostics = None
            else:
                diagnostics = engine.step(time["dt"], s["volume"], flow.get("shear_rate", 0.0))
            writer.writerow([step, step * time["dt"], ttr, trot, ttr / trot,
                             0 if diagnostics is None else diagnostics.candidates,
                             0 if diagnostics is None else diagnostics.accepted,
                             0 if diagnostics is None else diagnostics.infeasible_draws,
                             0 if diagnostics is None else diagnostics.maximum_energy_error,
                             0 if diagnostics is None else diagnostics.maximum_angular_momentum_error])


if __name__ == "__main__":
    main()

