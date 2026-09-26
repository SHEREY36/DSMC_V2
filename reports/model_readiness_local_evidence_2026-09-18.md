# Closure-v2 local evidence decision — 2026-09-18

## Decision

Do not release the fitted invariant corrections across flows.  Freeze the
schema-2.4 full-domain sampler artifact and run it with
`invariant_corrections: false` as the current good-enough baseline candidate.
No further excitation/refitting campaign is justified by the completed data.

The angular response is real and accurately predicts held-out angular laws,
but it leaves the calibrated feature domain in three of four exploratory USF
cases and has not yet been validated across flows.  The current USF DEM stress
reference is under repair and is not accepted as release evidence.  The energy
response is held back because it worsens conditional energy distributions.  A
response that is successful in isolation is not released before a trustworthy
cross-flow test exists.

## What the completed excitation campaign established

- 15,552 of 15,552 excitation observations completed; no execution failures.
- All 144 physical nodes have a full-rank, well-conditioned design.
- 141/144 nodes pass the central screening sentinel.
- The three suppressed nodes are `(alpha, theta, AR)` = `(0.5, 0.025, 1.1)`,
  `(1.0, 0.025, 1.1)`, and `(1.0, 0.1, 1.35)`.
- Held-out angular quantile W1 p95 improves from `0.03572` to `0.00846`.
- Central angular quantile W1 p95 improves from `0.01801` to `0.00224`.
- Held-out energy quantile W1 p95 worsens from `0.01783` to `0.03001` for the
  deployed tangent implementation (`0.02461` even for the exact fitted law).
- Central energy quantile W1 p95 worsens from `0.00759` to `0.00869` for the
  deployed tangent implementation.

The selective evidence compiler therefore releases zero energy rows, retains
angular rows at 141 nodes, and fully suppresses the three failed nodes.  This
is an auditable diagnostic candidate, not a deployable artifact.

## Why the HPC run did not create an NPZ

The array calculations and response fits completed.  The downstream artifact
pack job was protected by an `afterok` scientific gate; the excitation QA job
correctly returned failure because the response was not deployment-ready.
Consequently, packing never ran.  This was fail-closed behavior, not lost
compute.

A local evidence artifact has now been created without rebuilding the large
sampler tables:

- `models/microscopic_closure_v2_angular_evidence/closure_v2.npz`
- size: about 444 MB
- SHA-256: `f6a6cbec02ea83fa8bee0a5d5e80b575f5f1b8fc046d5f5cbcd098946ee3f616`
- status: `evidence_only_not_deployable`
- 144 coefficient nodes, 141 angular releases, zero energy releases.

The artifact is large because it preserves the validated adaptive energy
quantile tables and their lambda2/lambda3 sensitivities.  Only the small beta
tensors were replaced.  Recompressing those tensors cannot and should not turn
a 444 MB schema-2.4 artifact into the old 57 MB schema-2.3 representation.

## Reduced HCS evidence

The angular evidence artifact was run at the six independent DEM-backed HCS
nodes `(alpha, AR) = {0.8, 0.95, 1.0} x {2, 3}`, with two initial temperature
partitions and two realizations each.  All six pass the reduced physics gate;
mean closure overhead is 3.2–5.2%.

The same sampler with corrections disabled passes five cases at tau=12.  The
only miss, `(0.8, 3)`, exceeded the mean-drift cutoff by only 0.00183.  Extending
that case to tau=20 resolves the transient: mean theta is `0.9773` versus the
DEM target `1.0158`, initial-condition spread is `0.0136`, and maximum
replicate-mean drift is `0.0923`, so the case passes without correction.

This shows that the baseline HCS model is adequate at the six truth-backed
near-equipartition nodes; angular corrections are not required to obtain that
result.  A separate learned-extremes campaign now tests the near-sphere states
far from equipartition.

## Far-from-equipartition HCS evidence

The correction-disabled sampler was started at two exact learned low-theta
states, `theta0=0.0125` and `theta0=0.2`, for all
`alpha={0.5,0.8,0.95,1.0}` and `AR={1.1,1.2,1.35}`.  Each start used two
independent 1,000-particle realizations through `tau=30`.  Finite ensembles
were normalized to the declared modal temperatures, and a uniform HCS
similarity rescaling prevented numerical freezing without changing theta as a
function of collisions per particle.  Invariant corrections remained off.

All 48 low-side runs completed with zero out-of-domain queries, energy clamps,
monotonic repairs, or negative-energy repairs.  Seven of twelve cases already
meet the convergence/stationarity gate at `tau=30`:

| alpha | AR | late mean theta | start spread | max mean drift | pass at tau=30 |
|---:|---:|---:|---:|---:|:---:|
| 0.50 | 1.10 | 0.0532 | 0.2795 | 0.0704 | no |
| 0.50 | 1.20 | 0.1350 | 0.0346 | 0.1135 | no |
| 0.50 | 1.35 | 0.2544 | 0.0347 | 0.0377 | yes |
| 0.80 | 1.10 | 0.1929 | 0.0458 | 0.1571 | no |
| 0.80 | 1.20 | 0.3846 | 0.0352 | 0.0486 | yes |
| 0.80 | 1.35 | 0.6013 | 0.0152 | 0.0528 | yes |
| 0.95 | 1.10 | 0.3611 | 0.1109 | 0.1607 | no |
| 0.95 | 1.20 | 0.5419 | 0.0213 | 0.0830 | yes |
| 0.95 | 1.35 | 0.8170 | 0.0021 | 0.0468 | yes |
| 1.00 | 1.10 | 0.5843 | 0.2468 | 0.2241 | no |
| 1.00 | 1.20 | 0.9545 | 0.0677 | 0.0648 | yes |
| 1.00 | 1.35 | 1.0041 | 0.0248 | 0.0364 | yes |

The five misses are transients, not runtime-physics failures.  They are
concentrated at `AR=1.1`, where translational-rotational exchange is weakest,
plus `(0.5,1.2)`.  A dedicated 72-task Negishi campaign now reruns all twelve
cases with 2,000 particles, three replicas, and `tau=120` so the verdict is
uniform and not selected after looking at individual outcomes:

```bash
bash hpc/submit_hcs_learned_low.sh \
  models/microscopic_closure_v2_usf_candidate/closure_v2.npz learned_low_v1
```

An additional hot-boundary diagnostic found that elastic
`(alpha,AR,theta0)=(1,1.1,2)` can move infinitesimally above the maximum learned
theta after one stochastic collision.  The runtime correctly fails closed.
This must not be hidden by clamping or extrapolation; robust use of the exact
upper boundary would require calibrated support beyond theta=2.

## Reduced USF diagnostics (non-gating)

The user has identified problems in the LAMMPS/DEM USF stress output and is
repairing that reference.  The numbers below are retained only as a record of
the run and must not be used to accept or reject the closure.  The temperature
reference may be usable later, but it is not promoted to a gate until the
corrected reference dataset is frozen.

At `(alpha, AR) = {0.8, 0.95} x {2, 3}` with two realizations per arm:

| arm | stress relative RMSE | theta relative RMSE |
|---|---:|---:|
| corrections disabled | 3.58% | 1.89% |
| angular-only correction | 3.71% | 3.33% |

The apparent RMSE comparison is not a valid cross-flow verdict while the DEM
reference is uncertain.  Independently of that reference, the angular-only
response's feature-support diagnostics report out-of-domain fractions of
approximately 97% at `(0.8,2)`, 92% at `(0.8,3)`, 25% at `(0.95,2)`, and 0%
at `(0.95,3)`.  It must remain disabled in USF until support and reference
validation are both resolved.

The tau=80 uncorrected extension makes `(0.8,2)` stationary; its provisional
reference errors were 5.51% for stress and 4.62% for theta.  The near-elastic
`(0.95,2)` case is still warming at tau=80.  Extending it without changing the
model to tau=200 resolves the transient: provisional-reference stress and theta
errors are 2.21% and 2.23%, replicate-mean theta drift is 0.0064, and
total-temperature drift is 0.0661.  These are diagnostic values, not accuracy
claims against a trusted USF reference.

## Calibrated capability

The physical surface contains:

- alpha nodes: `0.5, 0.8, 0.95, 1.0`;
- aspect-ratio nodes: `1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0`;
- 144 valid `(alpha, theta, AR)` coordinates in total.

The runtime interpolates inside the sampled physical hull and fails closed
outside it.  Thus the artifact supports interpolation over `0.5 <= alpha <= 1`
and `1.1 <= AR <= 3` where the local theta coordinate is inside the calibrated
hull.  It does **not** presently claim `AR=1.0`; the spherical endpoint needs
an exact sphere branch or additional calibration and must not be inferred by
extrapolating the `AR=1.1` surface.

## Figures

- `reports/figures/full_candidate_excitation_validation.png`
- `reports/figures/hcs_angular_evidence_local.png`
- `reports/figures/hcs_baseline_evidence_local.png`
- `reports/figures/hcs_baseline_extended_local.png`
- `reports/figures/hcs_learned_low_local.png`
- `reports/figures/usf_angular_evidence_local.png`
- `reports/figures/usf_baseline_extended_local.png`
- `reports/figures/usf_baseline_near_elastic_local.png`
- `reports/figures/local_model_evidence_comparison.png`

## Remaining release work

1. Keep the full excitation dataset frozen; do not rerun or refit it.
2. Use the schema-2.4 sampler artifact with invariant corrections disabled.
3. Complete the correction-disabled learned-extremes and full-domain HCS
   validation, including the near-sphere nodes far from equipartition.
4. Hold USF release validation until the corrected LAMMPS/DEM reference is
   frozen.  Do not tune the model against the current stress data.
5. Add an explicit exact-sphere route if the promised domain must include
   `AR=1.0`; otherwise state the supported domain as `1.1 <= AR <= 3.0`.
6. A passing full-domain HCS gate is sufficient to begin a clearly labeled
   HCS-only non-Gaussian campaign.  General deployment across HCS and USF must
   still wait for the repaired USF reference and independent validation.
