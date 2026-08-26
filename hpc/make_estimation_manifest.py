#!/usr/bin/env python3
import argparse
import csv

from coll_models_v2.pipeline import discover_runs, group_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups = group_runs(discover_runs(args.runs_root))
    fields = ["task_id", "alpha", "theta", "aspect_ratio", "run_directories"]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for task, (key, paths) in enumerate(sorted(groups.items())):
            writer.writerow({"task_id": task, "alpha": key[0], "theta": key[1],
                             "aspect_ratio": key[2],
                             "run_directories": ";".join(str(path) for path in paths)})
    print(f"Wrote {len(groups)} estimation tasks")


if __name__ == "__main__":
    main()
