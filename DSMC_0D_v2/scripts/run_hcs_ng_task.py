#!/usr/bin/env python3
"""Execute one frozen-artifact non-Gaussian HCS realization."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_at(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= index < len(rows):
        raise IndexError(f"task {index} outside manifest with {len(rows)} rows")
    return rows[index]


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"expected true/false, got {value!r}")
    return normalized == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True, type=int)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--config",
                        default="DSMC_0D_v2/config/full_domain_baseline_candidate.yaml")
    args = parser.parse_args()
    row = row_at(Path(args.manifest), args.task)
    artifact = Path(args.artifact)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    # The submission preflight hashes the 444-MiB artifact once and embeds the
    # digest in every manifest row.  Rehashing it in hundreds of simultaneous
    # workers adds O(100 GiB) of shared-filesystem traffic without improving
    # the already fail-closed provenance check.
    artifact_hash = row.get("artifact_sha256") or sha256(artifact)
    config = yaml.safe_load(Path(args.config).read_text())
    alpha, ar = float(row["alpha"]), float(row["aspect_ratio"])
    sphere = row["arm"] == "sphere"
    model_variant = row.get("model_variant", "baseline")
    invariant_corrections = parse_bool(
        row.get("invariant_corrections", "false"))
    if model_variant == "angular_evidence" and not invariant_corrections:
        raise RuntimeError("angular-evidence task disabled its invariant correction")
    if model_variant in ("baseline", "sphere_exact") and invariant_corrections:
        raise RuntimeError(f"{model_variant} task unexpectedly enabled corrections")
    config["particle"]["AR"] = ar
    config["system"].update(alpha=alpha, kTt=1.0, kTr=1.0,
                            domain=[64.0, 64.0, 64.0])
    config.setdefault("simulation", {})["sphere_collision"] = sphere
    config["simulation"]["hcs_rescale_temperature"] = row["arm"] in ("scaled", "dt_half")
    if sphere:
        config["microscopic_closure"].update(
            routing="legacy_rank0", angular="legacy",
            invariant_corrections=False)
    else:
        config["microscopic_closure"].update(
            routing="variational_v2", angular="variational_v2",
            artifact=str(artifact), invariant_corrections=invariant_corrections,
            state_update_cpp=float(row.get("state_update_cpp", 0.0) or 0.0))
    params = particle_parameters(config)
    particles = int(row["particles"])
    volume = 64.0**3
    config["system"]["phi"] = (particles - 0.25) * params.volume / volume
    if math.ceil(config["system"]["phi"] * volume / params.volume) != particles:
        raise RuntimeError("failed to derive requested particle count")
    delta = float(row["sample_delta_tau"])
    config["time"].update(dt=float(row.get("dt") or 0.01), dtau=min(1.0, delta), t_end=100000.0,
                          tau_end=float(row["tau_end"]), equilibration_time=0.0)
    config["simulation"]["max_ntc_candidates_per_step"] = int(
        row.get("max_ntc_candidates_per_step") or max(100_000, 50 * particles))
    config["flow"] = {"mode": "hcs", "shear_rate": 0.0}
    config.setdefault("diagnostics", {})["collision_audit"] = True
    config["diagnostics"]["non_gaussian"] = {
        "enabled": True,
        "sample_start_tau": float(row["sample_start_tau"]),
        "sample_end_tau": float(row["sample_end_tau"]),
        "sample_delta_tau": delta,
        "minimum_tail_count": 1000,
        # Rods in this model are only weakly non-Gaussian (|a| <~ 0.07, a02<0
        # at small AR): beyond c=4 or w=6 even 5e7 particle-samples hold ~25
        # and 0 events, as for a Maxwellian.  These thresholds sit where the
        # data resolve thousands of events; fits there describe the
        # intermediate regime, not a proven asymptote.
        "c_tail_threshold": 3.0, "w_tail_threshold": 3.0,
        "x_tail_threshold": 20.0,
    }
    prefix = Path(row["output_prefix"])
    trajectory = Path(str(prefix) + ".txt")
    result = run_simulation(config, int(row["seed"]), trajectory)
    if not sphere and alpha < 1.0 and result.get("routing") != "variational_v2":
        raise RuntimeError("inelastic HCS-NG task did not run the frozen closure")
    if alpha >= 1.0 and not sphere and result.get("routing") != "elastic_bl":
        raise RuntimeError("elastic HCS-NG task did not run the exact elastic block")
    result["artifact"] = str(artifact)
    result["artifact_sha256"] = artifact_hash
    result["campaign"] = {
        key: (float(row[key]) if key in ("alpha", "aspect_ratio")
              else int(row[key]) if key in ("replicate", "seed", "particles")
              else row[key])
        for key in ("protocol_version", "mode", "arm", "model_variant",
                    "alpha", "aspect_ratio", "replicate", "seed", "particles")
        if key in row
    }
    result["campaign"]["invariant_corrections"] = invariant_corrections
    result["campaign"]["dt"] = float(config["time"]["dt"])
    output = Path(str(prefix) + ".json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
