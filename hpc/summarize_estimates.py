#!/usr/bin/env python3
"""Combine independently estimated node JSON files into grid QA tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected", type=int, default=870)
    args = parser.parse_args()

    sources = sorted(Path(args.input).glob("alpha_*.json"))
    if len(sources) != args.expected:
        raise ValueError(f"found {len(sources)} node estimates, expected {args.expected}")
    results = [json.loads(path.read_text()) for path in sources]
    results.sort(key=lambda row: (row["alpha"], row["theta"], row["aspect_ratio"]))
    keys = {(row["alpha"], row["theta"], row["aspect_ratio"]) for row in results}
    if len(keys) != args.expected:
        raise ValueError("node estimates contain duplicate parameter keys")
    for row in results:
        if "precision_pass" not in row.get("qa", {}):
            raise ValueError("node estimate lacks precision QA")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    long_fields = ["alpha", "theta", "aspect_ratio", "quantity", "estimate",
                   "standard_error", "ci_low", "ci_high"]
    with (output / "closure_coefficients_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for result in results:
            for name, quantity in result["quantities"].items():
                writer.writerow({"alpha": result["alpha"], "theta": result["theta"],
                                 "aspect_ratio": result["aspect_ratio"],
                                 "quantity": name, **quantity})

    qa_fields = ["alpha", "theta", "aspect_ratio", "n_attempts", "n_outcomes",
                 "precision_pass", "vss_representable", "continuation_reasons",
                 "cross_section_pass", "total_loss_compatibility_pass",
                 "score_tail_pass"]
    with (output / "qa_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=qa_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "alpha": result["alpha"], "theta": result["theta"],
                "aspect_ratio": result["aspect_ratio"],
                "n_attempts": result["n_attempts"], "n_outcomes": result["n_outcomes"],
                "precision_pass": result["qa"]["precision_pass"],
                "vss_representable": result["qa"]["vss_representable"],
                "continuation_reasons": ";".join(result["qa"]["continuation_reasons"]),
                "cross_section_pass": result["qa"]["cross_section_pass"],
                "total_loss_compatibility_pass": result["qa"]["total_loss_compatibility_pass"],
                "score_tail_pass": result["qa"]["score_tail_pass"],
            })
    passing = sum(bool(row["qa"]["precision_pass"]) for row in results)
    summary = {"schema_version": "2.1.0", "n_nodes": len(results),
               "n_pass": passing, "n_continue": len(results) - passing,
               "all_pass": passing == len(results),
               "audit": {
                   "cross_section_pass": sum(bool(row["qa"]["cross_section_pass"])
                                             for row in results),
                   "total_loss_compatibility_pass": sum(
                       bool(row["qa"]["total_loss_compatibility_pass"])
                       for row in results),
                   "vss_representable": sum(bool(row["qa"]["vss_representable"])
                                            for row in results),
               }}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summarized {len(results)} nodes: {passing} pass, "
          f"{len(results) - passing} require continuation")


if __name__ == "__main__":
    main()
