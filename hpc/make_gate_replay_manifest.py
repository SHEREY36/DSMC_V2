#!/usr/bin/env python3
"""Exact CTC replays of USF incoming pairs for the production gate.

Sources are DSMC-USF post-NTC collision-flux reservoirs (harvest windows).  Each
source is replayed at every requested alpha -- the fitted planes and points
between them -- so the gate compares the model with exact CTC on identical
pairs, both on the nodes and where the model interpolates.
"""
import argparse
import csv
import json
from pathlib import Path

NODES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
BETWEEN = [0.575, 0.675, 0.775, 0.875, 0.975, 0.99]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--harvest", default="results/usf_nonlinear_v2_20260928/harvest")
    parser.add_argument("--aspect-ratios", default="1.5,2,2.5,3")
    parser.add_argument("--source-alphas", default="0.5,0.8")
    parser.add_argument("--window", default="cold_replay_window_02")
    parser.add_argument("--alphas", default=",".join(map(str, NODES + BETWEEN)))
    parser.add_argument("--nsamples", type=int, default=40000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for ar in map(float, args.aspect_ratios.split(",")):
        for source_alpha in map(float, args.source_alphas.split(",")):
            stem = Path(args.harvest) / f"AR_{ar:.3f}_alpha_{source_alpha:.3f}_{args.window}"
            source = json.loads(Path(str(stem) + ".json").read_text())
            for alpha in map(float, args.alphas.split(",")):
                rows.append({
                    "task_id": len(rows), "alpha": alpha, "theta": source["theta"],
                    "aspect_ratio": ar, "source_alpha": source_alpha,
                    "seed": 290930000 + len(rows), "nsamples": args.nsamples,
                    "source_json": str(stem) + ".json", "replay_file": str(stem) + ".bin",
                    "output_directory": (f"results/production_gate_{args.tag}/replay/"
                                         f"AR_{ar:.3f}_src_{source_alpha:.3f}_alpha_{alpha:.3f}")})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"{args.output}: {len(rows)} replays")


if __name__ == "__main__":
    main()
