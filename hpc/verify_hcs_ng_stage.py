#!/usr/bin/env python3
"""Check one finished HCS-NG stage before spending the next allocation.

Every stage of the campaign is gated on the previous stage's summary, so the
cost of reading a summary wrongly is another multi-hour allocation.  This
prints the fields that decide the verdict and exits non-zero if the stage did
not pass, so the check is one command instead of a pasted assertion block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROTOCOL_VERSION = "hcs-ng-v8"
ANALYSIS_REVISION = "hcs-ng-analysis-v3"

# stage -> (mode, n_tasks, verdict field that authorizes the next stage)
STAGES = {
    "numerics-pilot": ("numerics-pilot", 24, "study_campaign_pass"),
    "engineering": ("engineering", 120, "study_campaign_pass"),
    "stability-sentinel": ("stability-sentinel", 80,
                           "long_time_stability_campaign_pass"),
    "stability": ("stability", 144, "long_time_stability_campaign_pass"),
    "sweep": ("sweep", 360, "scientific_outputs_released"),
}
REPORTED = (
    "protocol_version", "analysis_revision", "mode", "model_variant",
    "orientation_integrator", "artifact_sha256", "n_tasks",
    "n_completed_tasks", "n_cases", "terminal_sampling_recoveries",
    "physics_campaign_pass", "performance_campaign_pass",
    "engineering_control_coverage_pass", "two_sided_attraction_pass",
    "long_time_stability_campaign_pass", "study_campaign_pass",
    "scientific_outputs_released", "closure_domain_exclusions",
)


def failures(summary: dict, stage: str, artifact_sha256: str | None) -> list[str]:
    mode, n_tasks, verdict = STAGES[stage]
    problems = []
    if summary.get("protocol_version") != PROTOCOL_VERSION:
        problems.append(f"protocol_version is {summary.get('protocol_version')!r},"
                        f" expected {PROTOCOL_VERSION!r}")
    if summary.get("analysis_revision") != ANALYSIS_REVISION:
        problems.append(f"analysis_revision is {summary.get('analysis_revision')!r},"
                        f" expected {ANALYSIS_REVISION!r}")
    if summary.get("mode") != mode:
        problems.append(f"mode is {summary.get('mode')!r}, expected {mode!r}")
    if summary.get("n_tasks") != n_tasks:
        problems.append(f"n_tasks is {summary.get('n_tasks')}, expected {n_tasks}")
    if summary.get("n_completed_tasks") != n_tasks:
        problems.append(f"only {summary.get('n_completed_tasks')} of {n_tasks}"
                        " tasks completed")
    for field in ("failed_tasks", "missing_tasks"):
        if summary.get(field):
            problems.append(f"{field}: {summary[field][:10]}")
    if artifact_sha256 and summary.get("artifact_sha256") != artifact_sha256:
        problems.append("summary was produced from different artifact bytes")
    if not summary.get(verdict, False):
        problems.append(f"{verdict} is not true")
    for case in summary.get("cases") or ():
        reasons = [name for name in ("sampling_complete", "runtime_pass",
                                     "stationarity_pass",
                                     "dissipation_horizon_pass",
                                     "correction_fallback_pass")
                   if not case.get(name, True)]
        if reasons:
            problems.append(
                f"alpha={case['alpha']:.2f} AR={case['aspect_ratio']:.2f}"
                f" theta0={case.get('initial_theta')}: {', '.join(reasons)}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("summary")
    parser.add_argument("--artifact-sha256")
    args = parser.parse_args()
    path = Path(args.summary)
    if not path.is_file():
        raise SystemExit(f"no summary at {path}: the analysis job has not run")
    summary = json.loads(path.read_text())
    print(json.dumps({key: summary.get(key) for key in REPORTED}, indent=2))
    problems = failures(summary, args.stage, args.artifact_sha256)
    if problems:
        print(f"\nFAIL: {args.stage} did not pass\n", file=sys.stderr)
        for problem in problems[:40]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
        print(f"\nDo not submit the next stage. Full detail in {path}.",
              file=sys.stderr)
        raise SystemExit(1)
    print(f"\nPASS: {args.stage} is complete and authorizes the next stage")


if __name__ == "__main__":
    main()
