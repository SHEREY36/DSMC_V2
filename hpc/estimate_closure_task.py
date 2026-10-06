#!/usr/bin/env python3
"""Estimate one row of a schema-2.2 CTC campaign manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from coll_models_v2.estimate import (
    EXCHANGE_WEIGHTING,
    NODE_ESTIMATE_CONTRACT,
    equilibrium_anchor,
    estimate_node,
)
from coll_models_v2.pipeline import precision_status
from dsmc_v2_contracts import load_run, validate_run


SCIENTIFIC_FIT_EXCEPTIONS = (ValueError, np.linalg.LinAlgError, FloatingPointError)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _failed_fit_result(row: dict, run: Path, exc: Exception) -> dict:
    """Represent an expected sentinel model-form failure as gate evidence.

    The sentinel exists to discover infeasible projections and inadequate
    kernel forms.  Such a finding must block production, but it must not make
    Slurm treat the analysis as an infrastructure failure: the summary job
    still needs a result to report.
    """
    reason = f"scientific_fit_error:{type(exc).__name__}"
    return {
        "schema_version": "2.2.0",
        "estimator_contract": NODE_ESTIMATE_CONTRACT,
        "alpha": float(row["alpha"]),
        "theta": float(row["theta"]),
        "aspect_ratio": float(row["aspect_ratio"]),
        "ensemble_id": int(row["ensemble_id"]),
        "source_runs": [str(run.resolve())],
        "n_attempts": None,
        "n_outcomes": int(row["nsamples"]),
        "fit_error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "qa": {
            "propensity_pass": False,
            "proposal_balance_pass": False,
            "ess_pass": False,
            "energy_projection_pass": False,
            "angular_projection_pass": False,
            "model_form_pass": False,
            "elastic_pass": False,
            "sentinel_pass": False,
            "precision_pass": False,
            "continuation_reasons": [reason],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--propensity-offsets", type=int, default=128,
                        help="impact-plane samples for the kinematic acceptance "
                             "propensity; 0 falls back to the static 1/A_perp "
                             "weight, which is retained only for A/B runs")
    parser.add_argument("--propensity-workers", type=int, default=1,
                        help="threads used only for deterministic geometric "
                             "propensity blocks")
    parser.add_argument("--bootstrap-workers", type=int, default=1,
                        help="processes for the bootstrap refits (the replicate set is "
                             "identical for any count)")
    parser.add_argument("--skip-current", action="store_true",
                        help="leave a node alone when its estimate already carries the "
                             "current estimator contract, so a resubmitted array only "
                             "fits what is missing")
    parser.add_argument("--fallback-kernel-form", default=None,
                        help="refit with this energy-kernel form when the row's "
                             "form fails its precision QA (the repair used for "
                             "far-from-equilibrium near-sphere nodes)")
    args = parser.parse_args()
    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[args.index]
    output = Path(args.output)
    target = output / (
        f"alpha_{float(row['alpha']):.3f}_theta_{float(row['theta']):.3f}_"
        f"AR_{float(row['aspect_ratio']):.3f}_ensemble_{int(row['ensemble_id']):03d}.json")
    if args.skip_current and target.is_file():
        try:
            existing = json.loads(target.read_text())
        except ValueError:
            existing = {}
        if (existing.get("estimator_contract") == NODE_ESTIMATE_CONTRACT
                and existing.get("exchange_weighting") == EXCHANGE_WEIGHTING):
            print(f"task {args.index}: {target} is current; skipped")
            return
    run = Path(row["output_directory"])
    if not (run / "_SUCCESS").is_file():
        raise FileNotFoundError(f"closure shard is not finalized: {run}")

    # Missing, truncated, or internally inconsistent records remain hard job
    # failures.  Only failures encountered after contract validation are
    # converted into sentinel gate evidence.
    loaded = load_run(run)
    contract_qa = validate_run(loaded)
    if contract_qa["status"] != "pass":
        raise RuntimeError(
            "finalized closure shard failed contract validation: "
            + json.dumps(contract_qa, sort_keys=True)
        )
    # Every node at one aspect ratio shares ONE reference law: the elastic,
    # equipartitioned shard. Measuring it per node instead would declare each
    # node's own theta stationary and strip the kernel of its restoring force
    # everywhere except theta = 1. Each array task resolves it independently,
    # so no task can silently fall back to self-anchoring.
    # A canonical combined manifest can name one shared equilibrium shard for
    # every theta at an aspect ratio.  This prevents independently generated
    # duplicate anchors from injecting sampling noise into a reference law
    # that is mathematically theta-independent.  Older design manifests retain
    # the sibling-name fallback.
    anchor_text = row.get("anchor_directory", "").strip()
    anchor_run = Path(anchor_text) if anchor_text else run.with_name(
        re.sub(r"alpha_[0-9.]+_theta_[0-9.]+", "alpha_1.000_theta_1.000", run.name))
    if not anchor_run.is_dir():
        raise RuntimeError(
            f"cannot anchor: elastic equipartitioned shard missing at {anchor_run}")
    anchor = equilibrium_anchor([anchor_run], propensity_offsets=None)
    print(f"anchor for this aspect ratio: c1={anchor[0]:.5f} c2={anchor[1]:.5f}",
          flush=True)
    def fit(form: str):
        try:
            fitted = estimate_node([run], n_bootstrap=args.bootstrap,
                                   anchor=anchor,
                                   bootstrap_seed=20260902 + args.index,
                                   propensity_offsets=args.propensity_offsets or None,
                                   propensity_workers=args.propensity_workers,
                                   bootstrap_workers=args.bootstrap_workers,
                                   kernel_form=form)
        except SCIENTIFIC_FIT_EXCEPTIONS as exc:
            fitted = _failed_fit_result(row, run, exc)
            return fitted, False, fitted["qa"]["continuation_reasons"]
        ok, why = precision_status(fitted)
        fitted["qa"].update(precision_pass=ok, continuation_reasons=why)
        return fitted, ok, why

    kernel_form = row.get("kernel_form", "").strip() or "sinkhorn_bridge_v2"
    result, passed, reasons = fit(kernel_form)
    fallback = args.fallback_kernel_form
    if not passed and fallback and fallback != kernel_form:
        print(f"{kernel_form} failed QA ({', '.join(reasons)}); refitting with {fallback}",
              flush=True)
        retry, retry_passed, retry_reasons = fit(fallback)
        if retry_passed:
            retry["kernel_form_repair"] = {"primary": kernel_form,
                                           "primary_reasons": list(reasons)}
            result, passed, reasons, kernel_form = retry, True, retry_reasons, fallback
    result["estimator_provenance"] = {
        "git_sha": _git_sha(),
        "bootstrap": int(args.bootstrap),
        "propensity_offsets": int(args.propensity_offsets),
        "propensity_workers": int(args.propensity_workers),
        "manifest": str(Path(args.manifest)),
        "manifest_index": int(args.index),
        "anchor_run": str(anchor_run.resolve()),
        "kernel_form": result.get("energy", {}).get("kernel_form", kernel_form),
    }
    output.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = "pass" if passed else "blocked"
    print(f"task {args.index}: {target} ({status}; {', '.join(reasons) or 'no reasons'})")


if __name__ == "__main__":
    main()
