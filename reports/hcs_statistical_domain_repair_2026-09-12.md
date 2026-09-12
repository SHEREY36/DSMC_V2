# Corrected-HCS statistical-domain repair and continuation decision

## Outcome

The schema-2.4 candidate artifact passed construction and remained dynamically
stable in all 36 copied HCS trajectories.  The global `false` verdict was not a
physical instability.  It was caused by applying literal training-set minima
to six finite-particle U-statistics that estimate nonnegative population
squares.

The runtime and gate have been repaired without changing the collision model.
The next required high-compute operation is only a 36-task corrected-HCS
recheck.  The 1,296 excitation fits and the 1,440-node candidate artifact do
not need to be regenerated.

## Evidence from the completed campaign

All six case means were within 3.87% of their DEM targets.  Initial-partition
spread was at most 3.14%, replicate coefficient of variation at most 3.84%,
and replicate-mean late drift at most 5.39%.  Every one of those values is
inside the predeclared 10% HCS limits.  There were zero negative-energy
repairs, energy-axis clamps, and quantile-monotonicity repairs.

The 144 artifact-precompute stderr files, artifact aggregate stderr, 36 HCS
stderr files, and HCS plot stderr for jobs 43105268--43105271 are all empty.

The only systematic failure was an out-of-domain fraction of
0.9756--1.0000.  Fifteen trajectories also narrowly exceeded the independent
5% performance target; the campaign mean was 5.0097% and maximum 5.8754%.

## Statistical cause and repair

For a per-particle vector or tensor score `g_i`, the correction basis uses the
unbiased order-two U-statistic

\[
U_N = \frac{\left\|\sum_{i=1}^N g_i\right\|^2
             -\sum_{i=1}^N\|g_i\|^2}{N(N-1)}.
\]

It estimates the population quantity

\[
X=\|\mathbb E[g]\|^2\geq 0,
\]

but `U_N` may be negative when `X` is near zero.  The artifact minima were
only about 1e-7--1e-5 below zero, whereas normal 2,000-particle fluctuations
were about 1e-4--1e-3.  Consequently, `PiPi`, `QQ`, `RtRt`, `qtr2`, `qrot2`,
or `W2` marked nearly every runtime query as outside the box.

The correction still receives `U_N` exactly.  Domain classification now uses
the matching V-statistic only for these six one-sided quantities:

\[
V_N=\left\|\frac{1}{N}\sum_{i=1}^N g_i\right\|^2\geq 0.
\]

For the other eight signed features, the domain statistic equals the deployed
feature.  Thus the repair does not clip a feature, alter a coefficient, or
change a collision.  It asks whether the sample's physical moment is inside
the calibrated support while separately counting negative unbiased-estimator
excursions.

The implementation also computes `U_N` and `V_N` from the same reductions.
This avoids duplicated particle sums and is algebraically identical to the
previous U-statistic calculation.

## New diagnostics

Each DSMC result now records:

- `out_of_domain_fraction_by_feature`: genuine support violations;
- `sampling_excursion_fraction_by_feature`: raw U-statistic excursions that
  remain physically inside support;
- raw and domain-feature minima, means, and maxima when collision audit is on;
- `energy_monotonic_repairs` and the maximum repair in the aggregate HCS JSON.

A real-candidate short replay at `(alpha, theta0, AR)=(1,0.75,2)` produced:

- physical out-of-domain fraction: 0;
- negative-energy repairs: 0;
- energy-axis clamps: 0;
- quantile-monotonicity repairs: 0;
- closure overhead: 4.22%;
- complete runtime gate: pass.

The same replay retained large sampling-excursion fractions for the expected
near-zero one-sided estimators, proving that those events are visible rather
than silently discarded.

## Verification

- Focused domain/artifact tests: 22 passed.
- Full repository suite: 202 passed.
- Genuine upper support extrapolation remains a hard failure.
- Corrections-disabled behavior remains unchanged.
- The candidate NPZ is read-only in this repair and does not need rebuilding.

## Required Negishi stage

Run a fresh 36-trajectory candidate-HCS campaign into a new directory so the
old diagnostic evidence remains available.  The tasks are independent and
single-threaded, so 36 simultaneous one-core jobs are the accuracy-preserving
maximum useful parallelism.  Assigning the unused account cores to the same
Python processes would not shorten them.

```bash
git pull --ff-only origin closure-v2-repairs

bash hpc/submit_corrected_hcs_validation.sh \
  models/microscopic_closure_v2_candidate/closure_v2.npz \
  "" \
  manifests/hcs_corrected_validation_domain_v2.csv \
  results/hcs_corrected_validation_domain_v2
```

After the dependent plot job completes:

```bash
jq '{physics_gate_pass, production_gate_pass, cases}' \
  results/hcs_corrected_validation_domain_v2/summary.json

jq '[.runs[] | {
  task_id,
  out_of_domain_fraction,
  out_of_domain_fraction_by_feature,
  sampling_excursion_fraction_by_feature,
  energy_axis_clamps,
  energy_monotonic_repairs,
  negative_energy_repairs,
  closure_overhead_fraction
}]' results/hcs_corrected_validation_domain_v2/summary.json

ls -lh results/hcs_corrected_validation_domain_v2/hcs_theta_attraction.png
```

## Downstream gate

Do not rerun the excitation correction grid or rebuild the candidate while the
HCS recheck is pending.  After corrected HCS passes, the next scientific gate
is a small, independently generated direct-CTC sentinel campaign.  It is a
high-compute stage, but it must not yet be submitted: the current Fortran CTC
driver intentionally aborts for `ensemble_id != 0`, so direct nonequilibrium
initializers must be implemented and verified before spending cluster hours.

Only after that independent CTC comparison passes should the candidate be
promoted and USF validation begin.
