# Full-model campaign retry diagnosis — 2026-09-15

## Verdict

The campaign is not complete, but the completed scientific results remain usable.
The failure was in the Negishi Python environment, before the affected fits could
run; it was not a failed physics gate, optimizer rejection, or loss of numerical
accuracy.

## Evidence recovered locally

| Item | Count | Interpretation |
|---|---:|---|
| Designed missing-node fits | 11,664 | Six amplitudes, 18 response families, 108 physical nodes |
| Valid atomic fit outputs | 8,101 | Preserve and reuse |
| Explicitly blocked fit outputs | 0 | No scientific fit rejection |
| Invalid existing outputs | 0 | No corrupt published JSON result |
| Absent fit outputs | 3,563 | Retry set |
| Logs with the same SciPy import traceback | 3,561 | `CubicSpline` could not be imported |
| Absent outputs without a matching task traceback | 2 | Also included in the retry set |
| Support-refinement outputs | 0 of 1,296 | This dependent stage never started |

The common traceback is:

```text
ImportError: cannot import name 'CubicSpline' from 'scipy.interpolate'
    (unknown location)
```

The controller subsequently failed on the same import while validating artifact
inputs. No combined correction summary, coefficient surface, corrected artifact,
full-domain HCS result, fresh-CTC rescore, USF result, or non-Gaussian result was
created by this attempt.

Older non-empty errors in the copied log directory belong to superseded campaigns
(including old Slurm-spool path failures and the repaired `PosixPath` issue). They
must not be resubmitted: the current baseline validation already accepted all 144
baseline nodes.

## Repair

1. `hpc/verify_python_environment.py` now imports and executes the exact SciPy
   interpolation, spatial, and optimization components used by the pipeline.
2. The full pipeline runs that preflight before submitting any Slurm array.
3. `hpc/setup_negishi_env.sh` runs the same preflight after installation.
4. An explicitly selected `DSMC_V2_PYTHON` now fails closed instead of silently
   falling back to another interpreter.
5. A worker retries a pre-publication import/runtime failure up to three times;
   scientific blocked-result JSON remains a successful, auditable task result and
   is not disguised by retries.

Use a new, immutable environment rather than repairing `.conda-v2` in place. This
avoids changing shared packages underneath running workers.

## Exact continuation scope

Resume mode derives the retry manifest from absent output paths, so it submits only
the 3,563 missing fits and preserves the 8,101 completed fits byte-for-byte. It then
runs the previously unstarted 1,296 support-refinement fits. These 4,859 logical
observations are distributed over no more than 256 long-lived one-core worker-array
elements per manifest; they are not submitted as 4,859 simultaneous scheduler jobs.

After both arms finish, the combined design must contain 15,552 observations:

- 2,592 completed observations from the earlier correction-grid and USF-extension
  campaigns;
- 8,101 preserved observations from this campaign;
- 3,563 retried observations;
- 1,296 support-refinement observations.

Only then does the dependency chain compile the multivariate correction surface,
build the artifact, and run the full-domain HCS, independent CTC, USF, and
non-Gaussian gates. A gate failure will stop its dependent science campaign rather
than being treated as a scheduler crash.
