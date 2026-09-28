#!/usr/bin/env python3
"""Fit one exact replay law using its verified DSMC cell invariants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from coll_models_v2.estimate import estimate_node
from coll_models_v2.pipeline import precision_status
from dsmc_v2_contracts import FEATURE_NAMES, load_run, validate_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True,
                        help="index among replay rows, not all CTC rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--propensity-offsets", type=int, default=64)
    parser.add_argument("--propensity-workers", type=int, default=2)
    args = parser.parse_args()
    with Path(args.manifest).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["role"] == "usf_replay"]
    if not 0 <= args.index < len(rows):
        raise IndexError("replay estimator index is outside the manifest")
    row = rows[args.index]
    run_path = Path(row["output_directory"])
    if not (run_path / "_SUCCESS").is_file():
        raise FileNotFoundError(f"CTC input is not finalized: {run_path}")
    qa = validate_run(load_run(run_path))
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, sort_keys=True))
    run_metadata = load_run(run_path).metadata
    source_path = Path(row["source_metadata"]).resolve()
    source = json.loads(source_path.read_text())
    if (run_metadata.get("replay_source_metadata_sha256") is None
            or run_metadata.get("replay_source_binary_sha256")
            != source.get("binary_sha256")):
        raise RuntimeError("finalized CTC run lacks matching replay provenance")
    features = np.asarray([source["cell_features"][name]
                           for name in FEATURE_NAMES], dtype=float)
    artifact = Path(row["artifact"])
    artifact_sha256 = row["artifact_sha256"]
    with np.load(artifact, allow_pickle=False) as data:
        coordinates = np.asarray(data["surface_coordinates"], dtype=float)
        match = np.flatnonzero(np.all(np.isclose(
            coordinates, [1.0, 1.0, float(row["aspect_ratio"])],
            atol=1.0e-12, rtol=0.0), axis=1))
        if len(match) != 1:
            raise RuntimeError("base artifact lacks a unique equilibrium anchor")
        anchor = tuple(np.asarray(data["energy_anchor"], dtype=float)[match[0]])
    result = estimate_node(
        [run_path], n_bootstrap=args.bootstrap,
        bootstrap_seed=280904000 + args.index,
        propensity_offsets=args.propensity_offsets,
        propensity_workers=args.propensity_workers,
        anchor=anchor, cell_features_override=features,
        ensemble_id_override=int(row["ensemble_id"]),
        kernel_form="sinkhorn_bridge_v2")
    passed, reasons = precision_status(result)
    result["qa"].update(precision_pass=passed, continuation_reasons=reasons)
    result["direct_response_provenance"] = {
        "contract": "usf-direct-response-v1",
        "ctc_manifest": str(Path(args.manifest)),
        "replay_index": args.index,
        "source_metadata": str(source_path),
        "source_binary_sha256": source["binary_sha256"],
        "tau_window": source["tau_window"],
        "initial_branch": row["initial_branch"],
        "window_index": int(row["window_index"]),
        "anchor_source": "frozen_base_artifact_equilibrium_node",
        "base_artifact": str(artifact.resolve()),
        "base_artifact_sha256": artifact_sha256,
        "equilibrium_anchor": list(map(float, anchor)),
        "dem_used_as_fit_target": False,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"replay_{args.index:03d}.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {target}; precision_pass={passed}")


if __name__ == "__main__":
    main()
