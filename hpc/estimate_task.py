#!/usr/bin/env python3
"""Decode and estimate one fixed-grid node from pilot or all completed shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coll_models_v2.estimate import estimate_node
from coll_models_v2.legacy_bl import LegacyBL
from coll_models_v2.pipeline import precision_status

from make_manifest import base_rows


def finalized_path(path: Path) -> Path | None:
    """Resolve a completed shard, including the pre-v2.1.1 CRLF path bug."""
    if (path / "_SUCCESS").is_file():
        return path
    legacy = Path(str(path) + "\r")
    if (legacy / "_SUCCESS").is_file():
        return legacy
    return None


def node_paths(index: int, scope: str, runs_root: Path) -> tuple[dict, list[Path]]:
    pilot_rows = base_rows("pilot", 20_000, 0, str(runs_root))
    production_rows = base_rows("production", 80_000, 1, str(runs_root))
    if index < 0 or index >= len(pilot_rows):
        raise IndexError(f"task index {index} lies outside 0..{len(pilot_rows) - 1}")
    row = pilot_rows[index]
    expected = [Path(row["output_directory"])]
    if scope == "combined":
        expected.append(Path(production_rows[index]["output_directory"]))
    resolved = [finalized_path(path) for path in expected]
    missing = [path for path, actual in zip(expected, resolved) if actual is None]
    if missing:
        raise FileNotFoundError("required finalized shard(s) missing: "
                                + ", ".join(map(str, missing)))
    paths = [path for path in resolved if path is not None]
    if scope == "combined":
        pattern = (f"alpha_{float(row['alpha']):.3f}_theta_{float(row['theta']):.3f}_"
                   f"AR_{float(row['aspect_ratio']):.3f}_shard_*")
        paths.extend(sorted(path for path in (runs_root / "continuation").glob(pattern)
                            if (path / "_SUCCESS").is_file()))
    return row, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--scope", choices=("pilot", "combined"), required=True)
    parser.add_argument("--runs-root", default="results/ctc")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--gamma-max-table", required=True)
    parser.add_argument("--one-hit-table", required=True)
    args = parser.parse_args()

    row, paths = node_paths(args.index, args.scope, Path(args.runs_root).resolve())
    bl = LegacyBL.load(args.gamma_max_table, args.one_hit_table)
    result = estimate_node(paths, bl, n_bootstrap=args.bootstrap)
    passed, reasons = precision_status(result)
    result["qa"].update(precision_pass=passed, continuation_reasons=reasons)
    result["source_scope"] = args.scope
    result["source_runs"] = [str(path) for path in paths]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / (
        f"alpha_{float(row['alpha']):.3f}_theta_{float(row['theta']):.3f}_"
        f"AR_{float(row['aspect_ratio']):.3f}.json")
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"{args.scope} task {args.index}: {target} "
          f"({'pass' if passed else 'continue'}) from {len(paths)} shard(s)")


if __name__ == "__main__":
    main()
