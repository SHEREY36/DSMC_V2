#!/usr/bin/env python3
"""Fail closed when a scientific HCS-NG design outruns artifact evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--hcs-summary")
    parser.add_argument("--pilot-summary",
                        help="passing engineering-pilot summary; gates the sweep")
    parser.add_argument("--allow-engineering", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.manifest).open(newline="")))
    if not rows:
        raise SystemExit("empty non-Gaussian manifest")
    modes = {row["mode"] for row in rows}
    if len(modes) != 1:
        raise SystemExit("manifest mixes campaign modes")
    mode = modes.pop()
    artifact = Path(args.artifact)
    artifact_hash = digest(artifact)
    if mode == "engineering":
        if not args.allow_engineering:
            raise SystemExit("engineering mode requires --allow-engineering")
    elif mode == "sweep":
        # The sweep is an HCS-only campaign on the frozen baseline.  It is
        # gated on the engineering pilot of the same bytes: stationarity,
        # scaled/unscaled equivalence and zero runtime repairs.
        if not args.pilot_summary:
            raise SystemExit("sweep requires --pilot-summary from a passing engineering pilot")
        pilot = json.loads(Path(args.pilot_summary).read_text())
        if pilot.get("mode") != "engineering":
            raise SystemExit("pilot summary is not an engineering-mode summary")
        if not pilot.get("physics_campaign_pass", False):
            raise SystemExit("engineering pilot physics verdict has not passed")
        if pilot.get("artifact_sha256") != artifact_hash:
            raise SystemExit("engineering pilot did not use these artifact bytes")
    elif mode != "sphere-controls":
        if not args.hcs_summary:
            raise SystemExit("scientific campaign requires --hcs-summary")
        hcs = json.loads(Path(args.hcs_summary).read_text())
        if not hcs.get("full_domain_physics_gate_pass", False):
            raise SystemExit("expanded full-domain HCS physics gate has not passed")
        if hcs.get("artifact_sha256") != artifact_hash:
            raise SystemExit("expanded HCS gate did not use these artifact bytes")

    if mode != "sphere-controls":
        # Invariant corrections are off, so the governing support is the
        # sampler surface, not the correction (beta) hull.
        with np.load(artifact, allow_pickle=False) as data:
            points = np.asarray(data["surface_coordinates"], dtype=float)
        if len(points) < 4:
            raise SystemExit("artifact has no interpolable sampler surface")
        hull = Delaunay(points)
        unsupported = []
        for row in rows:
            if float(row["alpha"]) >= 1.0:
                continue   # exact elastic block; never queries the artifact
            # Trajectories start at theta=1 and move toward theta_H; the
            # runtime still fails closed if one actually leaves the hull.
            for theta in (1.0,):
                query = [float(row["alpha"]), theta, float(row["aspect_ratio"])]
                if hull.find_simplex(query) < 0 and not np.any(np.all(
                        np.isclose(points, query, atol=1e-12), axis=1)):
                    unsupported.append(query); break
        if unsupported:
            preview = unsupported[:5]
            raise SystemExit(f"artifact sampler hull misses campaign cases {preview}"
                             f" ({len(unsupported)} rows unsupported)")
    print(f"HCS-NG prerequisites pass for {mode}: {len(rows)} tasks; artifact={artifact_hash}")


if __name__ == "__main__":
    main()
