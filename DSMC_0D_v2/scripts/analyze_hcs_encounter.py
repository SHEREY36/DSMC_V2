#!/usr/bin/env python3
"""Late-time HCS temperature ratio per arm versus the DEM benchmark."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-start-tau", type=float, default=40.0)
    args = parser.parse_args()
    rows = []
    for row in csv.DictReader(open(args.manifest)):
        path = Path(row["output_prefix"] + ".txt")
        if not path.is_file() or not Path(row["output_prefix"] + ".json").is_file():
            continue
        t = np.loadtxt(path)
        late = t[:, 1] >= args.window_start_tau
        rows.append({"arm": row["arm"], "alpha": float(row["alpha"]), "AR": float(row["aspect_ratio"]),
                     "theta0": float(row["theta0"]), "replicate": int(row["replicate"]),
                     "target": float(row["target_theta"]),
                     "theta": float(np.mean(t[late, 2] / t[late, 3]))})
    runs = pd.DataFrame(rows)
    case = runs.groupby(["arm", "alpha", "AR"]).agg(theta=("theta", "mean"), sem=("theta", "sem"),
                                                    target=("target", "first"),
                                                    start_gap=("theta", lambda x: float(np.ptp(x))),
                                                    n=("theta", "size")).reset_index()
    case["vs_DEM"] = case.theta / case.target - 1.0
    case.to_csv(args.output, index=False)
    print(case.round(4).to_string())
    Path(args.output).with_suffix(".json").write_text(json.dumps(
        {arm: {"max_abs_vs_DEM": float(np.max(np.abs(g.vs_DEM)))} for arm, g in case.groupby("arm")}, indent=2))


if __name__ == "__main__":
    main()
