#!/usr/bin/env python3
"""Run one realization of the encounter-unit USF campaign (usf-encounter-v1)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

from dsmc_v2.particle import particle_parameters
from dsmc_v2.simulation import run_simulation

ARMS = {
    "contact_legacy": {"event_unit": "contact_legacy", "memory": False},
    "encounter": {"event_unit": "encounter", "memory": False},
    "encounter_memory": {"event_unit": "encounter", "memory": True},
}


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--config", default="DSMC_0D_v2/config/full_domain_baseline_candidate.yaml")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--encounter-table",
                        default="DSMC_0D_v2/models/encounter_cross_section_v2.json")
    parser.add_argument("--angular-memory", default="DSMC_0D_v2/models/angular_memory_v1.json")
    args = parser.parse_args()
    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[args.task]
    if int(row["task_id"]) != args.task or row["protocol_version"] != "usf-encounter-v1":
        raise ValueError("manifest row does not match the requested task")
    prefix = Path(row["output_prefix"])
    done = Path(str(prefix) + ".json")
    if done.is_file():
        print(f"already complete: {done}")
        return
    arm = ARMS[row["arm"]]

    config = yaml.safe_load(Path(args.config).read_text())
    ar, alpha = float(row["aspect_ratio"]), float(row["alpha"])
    particles, phi = int(row["particles"]), float(row["volume_fraction"])
    config["particle"]["AR"] = ar
    config["system"].update({"alpha": alpha, "kTt": float(row["initial_ttr"]),
                             "kTr": float(row["initial_trot"]), "phi": phi,
                             "particle_count": particles})
    params = particle_parameters(config)
    box = (particles * params.volume / phi) ** (1.0 / 3.0)
    config["system"]["domain"] = [box, box, box]
    if not math.isclose(particles * params.volume / box**3, phi, rel_tol=2e-14, abs_tol=2e-14):
        raise RuntimeError("failed to construct the requested volume fraction")
    config["time"].update({"dt": float(row["dt"]), "dtau": 0.5, "t_end": 1.0e9,
                           "tau_end": float(row["tau_end"]), "equilibration_time": 0.0})
    config["flow"] = {"mode": "usf", "shear_rate": float(row["shear_rate"])}
    config["simulation"].update({"sphere_collision": False, "exact_initial_temperatures": True,
                                 "orientation_integrator": "symmetric_midpoint_v1",
                                 "use_isotropic_eps": True})
    closure = {"routing": "variational_v2", "angular": "variational_v2",
               "artifact": args.artifact, "invariant_corrections": False,
               "state_update_cpp": float(row["state_update_cpp"]),
               "correction_fallback_gate": "adaptive_base_law",
               "event_unit": arm["event_unit"],
               "encounter_cross_section": args.encounter_table}
    if arm["memory"]:
        closure["angular_memory"] = args.angular_memory
    config["microscopic_closure"].update(closure)
    config.setdefault("diagnostics", {})["collision_audit"] = False

    prefix.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = run_simulation(
        config, int(row["seed"]), Path(str(prefix) + ".txt"),
        Path(str(prefix) + "_pressure.txt"), Path(str(prefix) + "_orientation.txt"))
    provenance = {"artifact": args.artifact, "artifact_sha256": digest(args.artifact),
                  "encounter_table_sha256": digest(args.encounter_table)}
    if arm["memory"]:
        provenance["angular_memory_sha256"] = digest(args.angular_memory)
    diagnostics.update({
        "protocol_version": "usf-encounter-v1", "provenance": provenance,
        "validation_case": {key: row[key] for key in row},
        "particle_mass": params.mass, "particle_diameter": params.diameter,
    })
    temporary = done.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(diagnostics, indent=2, sort_keys=True, default=float) + "\n")
    temporary.replace(done)


if __name__ == "__main__":
    main()
