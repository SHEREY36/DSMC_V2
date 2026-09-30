#!/usr/bin/env python3
"""Late-time HCS temperature ratio per arm versus the DEM benchmark (no pandas)."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-start-tau", type=float, default=40.0)
    args = parser.parse_args()
    groups = defaultdict(list)
    for row in csv.DictReader(open(args.manifest)):
        path = Path(row["output_prefix"] + ".txt")
        if not path.is_file() or not Path(row["output_prefix"] + ".json").is_file():
            continue
        t = np.loadtxt(path)
        late = t[:, 1] >= args.window_start_tau
        groups[(row["arm"], float(row["alpha"]), float(row["aspect_ratio"]))].append(
            (float(np.mean(t[late, 2] / t[late, 3])), float(row["target_theta"])))
    rows = []
    for (arm, alpha, ar), vals in sorted(groups.items()):
        theta = np.array([v[0] for v in vals]); target = vals[0][1]
        rows.append({"arm": arm, "alpha": alpha, "AR": ar, "theta": float(theta.mean()),
                     "sem": float(theta.std(ddof=1) / np.sqrt(len(theta))) if len(theta) > 1 else float("nan"),
                     "start_gap": float(np.ptp(theta)), "n": len(theta), "target": target,
                     "vs_DEM": float(theta.mean() / target - 1.0)})
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for r in rows:
        print(f"{r['arm']:18s} alpha={r['alpha']:.2f} AR={r['AR']:.1f} theta={r['theta']:.4f} "
              f"DEM={r['target']:.4f} ({100 * r['vs_DEM']:+.2f}%)")
    Path(args.output).with_suffix(".json").write_text(json.dumps(
        {arm: {"max_abs_vs_DEM": max(abs(r["vs_DEM"]) for r in rows if r["arm"] == arm)}
         for arm in {r["arm"] for r in rows}}, indent=2))


if __name__ == "__main__":
    main()
