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
    elif mode != "sphere-controls":
        if not args.hcs_summary:
            raise SystemExit("scientific campaign requires --hcs-summary")
        hcs = json.loads(Path(args.hcs_summary).read_text())
        if not hcs.get("full_domain_physics_gate_pass", False):
            raise SystemExit("expanded full-domain HCS physics gate has not passed")
        if hcs.get("artifact_sha256") != artifact_hash:
            raise SystemExit("expanded HCS gate did not use these artifact bytes")

    if mode != "sphere-controls":
        with np.load(artifact, allow_pickle=False) as data:
            points = np.asarray(data["beta_coordinates"], dtype=float)
        if len(points) < 4:
            raise SystemExit("artifact has no interpolable correction surface")
        hull = Delaunay(points)
        unsupported = []
        for row in rows:
            # The trajectory starts at theta=1.  Testing 0.2 and 2 as well
            # requires the correction surface to bracket ordinary HCS motion;
            # runtime still fails closed if an actual trajectory goes farther.
            for theta in (0.2, 1.0, 2.0):
                query = [float(row["alpha"]), theta, float(row["aspect_ratio"])]
                if hull.find_simplex(query) < 0 and not np.any(np.all(
                        np.isclose(points, query, atol=1e-12), axis=1)):
                    unsupported.append(query); break
        if unsupported:
            preview = unsupported[:5]
            raise SystemExit(f"artifact correction hull misses campaign cases {preview}"
                             f" ({len(unsupported)} rows unsupported)")
    print(f"HCS-NG prerequisites pass for {mode}: {len(rows)} tasks; artifact={artifact_hash}")


if __name__ == "__main__":
    main()
