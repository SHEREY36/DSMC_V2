# HCS-NG protocol v8 - 2026-09-25

Supersedes `HCS_NG_PROTOCOL_V7_2026-09-23.md`. v8 is a measurement-design
repair, not a physics change. The v7 148-task two-sided stability campaign
completed all 148 tasks and passed all 37 two-sided attraction tests; its
`physics_campaign_pass: false` came from three unrelated defects in the
instrumentation around that result.

## What the v7 stability campaign actually established

Both initial branches reach a common HCS attractor at every coordinate. That
is the question the stage exists to answer and it came back clean. The
campaign also produced, for the first time, a direct measurement of how long
the attractor takes to form.

## Defect 1: a hard-coded physical-time ceiling truncated runs silently

`run_hcs_ng_task.py` set `t_end = 100000.0` while the manifest asked for
`tau_end = 6153.85` at alpha=0.95. The march loop terminates on
`time < end_time`, and nothing recorded which clock stopped it, so
`run_status` stayed `complete`.

| coordinate | requested cpp | reached cpp | final t |
|---|---|---|---|
| alpha=0.95, AR=1.10 | 6153.8 | 4872 | 99991 |
| alpha=0.95, AR=1.20 | 6153.8 | 6035 | 99999 |
| alpha=0.95, AR=3.00 | 6153.8 | 6153 | 23103 |

At fixed box volume a near-sphere rod collides about five times less often
per unit time than a long one, so the same collision target costs it about
five times more physical time. Four tasks burned 16.4 h each and were
discarded; the failure surfaced four gates downstream.

**Repair.** `run_simulation` reports `termination_reason`, `final_time`,
`requested_tau_end` and `requested_t_end`. The HCS-NG runner marches on the
collision clock alone (`t_end = inf`) and raises if the run stops on anything
else; the analyzer rejects such a record. A stalled march is now bounded by
the scheduler and appears as a missing task, which fails closed.

## Defect 2: the stationarity gate estimated its tolerance from two replicates

`replicate_stationarity` formed the drift standard error from two values -
one degree of freedom - and multiplied it by 3.0 as though it were a normal
quantile. Across statistically identical v7 coordinates that estimate ranged
from 0.0002 to 0.033. A tight pair collapsed the tolerance and failed a quiet
run as drift; a loose pair inflated it past the ceiling and failed the same
run as under-resolved. Measured false-rejection rate at the v7 design:
**14.7 per cent per observable**, which is why the campaign failed a
different random subset of coordinates each time it ran.

Underneath that the measurement was under-powered. Per-snapshot scatter at
N=2000 was sd(a20) = 0.016-0.065 against a declared 0.01 drift tolerance.

**Repair.** v3 analysis estimates the drift as an OLS trend across the
retained window and takes its variance from the **detrended** residual with an
autocorrelation correction, pooled across replicates. Detrending matters: the
raw series would let a genuine drift inflate its own error bar and pass
itself. That estimator carries of order `n_samples / tau_int` degrees of
freedom per replicate instead of one, so the 3.0 multiplier is legitimate.
The between-replicate spread is retained as a reported cross-check.

Measured behaviour of the repaired gate:

| design | stationary coordinate rejected | real 0.02 drift caught |
|---|---|---|
| v7: N=2000, 40 samples | 14.7 % | yes |
| v8: N=10000, 201 samples | 0.0 % | 100 % |

The gate is not softened. It is made resolvable, and it keeps full power
against a real drift.

## Defect 3: the horizon rode the wrong clock

v7 required a common accumulated cooling `chi = (1-alpha^2)*tau >= 600`,
giving `tau_end = 6154` at alpha=0.95.

The v7 trajectories show both branches enter the late-time band within
**tau <= 130 collisions per particle** at all 37 coordinates (median 21, p90
62, worst 130 at alpha=0.95, AR=1.1). In `chi` units the same measurement
spans only 6-41. Relaxation of the modal temperature ratio is collisional
energy exchange between modes, so it lives on the collision clock and is
nearly flat in alpha; the cooling clock is the wrong variable. The v7 rule
over-ran the slowest coordinate by a factor of about 47.

**Repair.** Every long-time stage runs the production window,
`tau in [500, 1500]` with `delta tau = 5` - which is also the window of
Megias & Santos 2023 - giving at least 3.8 relaxation times before the first
retained snapshot at the worst coordinate and 24 at the median. `chi` is
still reported; it no longer sets the run length.

## Statistics are nearly free, and v7 declined them

Wall time per task is set by the collision count and the aspect ratio, not by
the particle count. At fixed box volume the number density scales with N, so
the physical time to reach a given cpp falls as 1/N while the per-step cost
rises as N. Measured across five coordinates (same alpha, AR, arm, dt):

| coordinate | s/cpp at N=2000 | s/cpp at N=10000 | ratio |
|---|---|---|---|
| alpha=0.50, AR=1.35 | 10.21 | 11.35 | 1.11 |
| alpha=0.50, AR=3.00 | 2.82 | 4.33 | 1.54 |
| alpha=0.80, AR=2.00 | 4.56 | 5.03 | 1.10 |
| alpha=0.95, AR=2.00 | 4.16 | 5.48 | 1.32 |
| alpha=1.00, AR=3.00 | 2.02 | 2.33 | 1.15 |

N=10000 costs about 1.25x the wall time of N=2000 for the same cpp and gives
five times the samples per snapshot, a 2.24x gain in resolution. The v7
screen ran N=2000 and bought nothing with it. Every v8 stage runs N=10000.

Peak RSS is artifact-dominated (645 MiB at both N), so 1500M per task stands.

## Design

| stage | tasks | grid | starts x seeds | N | tau_end | worst task |
|---|---|---|---|---|---|---|
| numerics-pilot | 24 | 1 | 1 x 8, 3 arms | 10000 | 60 | ~0.6 h |
| engineering | 120 | 5 | 1 x 8, 3 arms | 10000 | 60 | ~2.1 h |
| stability-sentinel | 80 | 5 | 2 x 4, 2 arms | 10000 | 1500 | ~11 h |
| stability | 144 | 36 | 2 x 2 | 10000 | 1500 | ~8.4 h |
| sweep | 360 | 36 | 2 x 5 | 10000 | 1500 | ~8.4 h |

The production sweep is now two-sided: five realizations from each bracketing
start rather than ten from `theta0 = 1`. The attraction test therefore rests
on production statistics, and once the branches are shown to agree all ten
realizations pool for the cumulants and the tail fits at no statistical cost.
Gates stay on the per-branch cases; the pooled `science_cases` carry the
figure. Because the sweep now starts at `theta0 = 0.025` for AR <= 1.35 it
depends on the sampler's low-theta extension, which the one-sided design did
not; the preflight fails closed if an artifact lacks it.

Total: about 513 core-hours for stability (against 609 already spent on the
failed v7 screen) and about 1250 for the sweep.

## Declared model-domain exclusion: alpha=0.50, AR=1.10

The HCS attractor there sits at a20 ~ 0.140 and Trot/Ttr ~ 20, while the
artifact's `a2_tr` support is [-0.086, +0.117]. Forty-nine per cent of
evaluation-window closure queries fell back to the base law, so that
coordinate measures the fallback, not the learned closure. This is a real
boundary: the near-sphere strongly-inelastic corner approaches the
smooth-sphere limit where rotation decouples and theta runs away. The
coordinate is excluded from the production grid and reported as the model's
validated-domain boundary.

AR=1.20 stays in. Its mean a20 = 0.055 is well inside the hull; its 3-5 per
cent v7 fallback was an N=2000 boundary excursion (SE(a20) = 0.028 at N=2000
against 0.013 at N=10000, so a +2.2 sigma event becomes +4.9 sigma). The v8
particle count is expected to remove it, and `closure_domain_exclusions` in
the summary will confirm or refute that.

## Re-analysis of the v7 data under analysis v3

For the record, without rerunning anything:

- `two_sided_attraction_pass: true` at all 37 coordinates;
- 415 of 444 observable-gates pass;
- 26 failures are `late_window_drift_under_resolved` or `too_few_late_samples`
  - the gate correctly reporting that N=2000 cannot test a 0.01 drift, and the
  four truncated tasks;
- only 3 are resolved drifts, all within 1.0-1.3x of their own 3-sigma
  resolution, to be settled at production statistics.

There is no evidence in the v7 data of a non-stationary HCS or of a
timestep-dependent attractor.

## Submission order

`numerics-pilot -> engineering -> stability-sentinel -> stability -> sweep`,
each gated on the preceding passing summary, on identical artifact bytes, on
`protocol_version: hcs-ng-v8` and on `analysis_revision: hcs-ng-analysis-v3`.
v7 summaries cannot unlock a v8 stage.
