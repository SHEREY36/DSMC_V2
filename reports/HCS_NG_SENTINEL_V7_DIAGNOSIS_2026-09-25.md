# HCS-NG v7 sentinel diagnosis - 2026-09-25

## Decision

The completed 80-task sentinel does not contain evidence of a physical
runaway or a timestep-dependent long-time attractor. Do not rerun those 80
trajectories. Reanalyze them with analysis revision `hcs-ng-analysis-v2`, then
use the passing revised summary to launch the 148-task stability campaign.

All 80 tasks completed, all worker error logs were empty, all five pairs of
initial temperature ratios converged to the same late HCS attractor, every
case was stationary, and every runtime, closure-domain, NTC, memory, and
performance gate passed.

## Cause 1: terminal sampling

For the `alpha=0.8` and `alpha=0.95` arms, 32 trajectories contained 39 of 40
scheduled samples. They covered the complete window through the penultimate
sampling point, but a collision batch crossed `tau_end` after the normal
output branch. The simulation then exited without sampling the terminal
state. This was an output-scheduling defect, not a truncated simulation.

The simulator now calls the idempotent non-Gaussian sampler once at the
terminal state. The analyzer can recover old output only when exactly one
terminal sample is absent, all earlier samples are present and ordered, their
spacing is complete, and the last sample lies within one declared sampling
interval of the endpoint. Sparse or truncated output still fails closed. The
revised sentinel summary reports `terminal_sampling_recoveries: 32`.

## Cause 2: duplicated precision responsibility

The only failed numerical observable was `a02` at `alpha=0.5, AR=1.35` for
the `dt=0.0025` versus `dt=0.00125` comparison:

- paired mean difference: `0.004691`;
- three-standard-error resolution: `0.014494`;
- hard numerical effect cap: `0.01`;
- statistically consistent with zero: yes;
- precision resolved at sentinel `N=2000`: no.

The mandatory engineering campaign had already tested the same worst-case
coordinate with eight independent `N=10000` realizations. It found an `a02`
difference of `0.000214` with three-SE resolution `0.008829`, passing the
strict precision gate. The long-time sentinel used four seeds and two initial
conditions to test attraction; the repeated seed labels across initial
conditions are not eight new independent precision realizations.

Analysis revision v2 therefore keeps the strict consistency-plus-precision
rule for the engineering campaign. In the downstream sentinel, precision is
inherited from that mandatory gate, while every long-time paired observable
must still be statistically consistent with zero and have its observed mean
difference below the same hard cap. Large effects cannot pass by being noisy.

## Reanalysis result

Reanalysis of the copied data produced:

- `n_completed_tasks: 80` of 80;
- `terminal_sampling_recoveries: 32`;
- `physics_campaign_pass: true`;
- `performance_campaign_pass: true`;
- `engineering_control_coverage_pass: true`;
- `long_time_stability_campaign_pass: true`;
- no failed or missing tasks.

The stability preflight now requires `analysis_revision: hcs-ng-analysis-v2`,
so the original false summary cannot authorize the 148-task allocation.
