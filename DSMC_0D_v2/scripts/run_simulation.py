#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dsmc_v2.simulation import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conservative DSMC v2.1 HCS")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--pressure-output")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(args.seed if args.seed is not None else config["simulation"]["seeds"][0])
    output = args.output or str(Path(config["simulation"]["output_dir"]) / "hcs_v2.txt")
    print(json.dumps(run_simulation(config, seed, output, args.pressure_output),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
