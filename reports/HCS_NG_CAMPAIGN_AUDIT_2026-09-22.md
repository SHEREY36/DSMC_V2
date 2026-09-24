# HCS non-Gaussian campaign audit - 2026-09-22

> **Superseded on 2026-09-23.** The completed protocol-v3 sweep exposed a
> delayed apparent HCS energy-partition instability after its short engineering
> gate. The cause was later identified as similarity amplification of
> centre-of-mass roundoff that the old code incorrectly included in `Ttr`, not
> as a demonstrated instability of the energy kernel. Its inelastic
> measurements remain diagnostic because the contaminated ratio was fed back
> into the closure. Do not use the v3 submission commands below. Protocol v4
> and the current Negishi sequence are documented in
> `reports/HCS_NG_PROTOCOL_V5_2026-09-23.md`.

## Decision

The failed `hcs_ng_sweep_baseline_v1` allocation must not be resumed or used as
the scientific campaign. It used the older sampler artifact with invariant
corrections disabled, protocol-v1 statistics, and a rescaling bug that made the
NTC majorant grow monotonically. Of 290 submitted tasks, 100 completed and 190
did not produce final results. The completed tasks are useful forensic data,
but they do not have the artifact or protocol requested for the HCS study.

The replacement is protocol `hcs-ng-v3`:

- use `models/microscopic_closure_v2_angular_evidence/closure_v2.npz`;
- enable only its validated angular response (`validated-angular-only-v1`);
- keep all energy-response rows disabled, as encoded in that artifact;
- use the exact Hong-Morris elastic block at `alpha=1` instead of querying the
  learned closure;
- validate rescaling and time-step convergence in a 120-task engineering gate;
- only then submit the 370-task scientific sweep.

This is an HCS evidence campaign. The angular artifact remains
`evidence_only_not_deployable`; success here must not be described as USF or
cross-flow validation.

Protocol v2 completed all 120 engineering tasks without execution errors, but
correctly blocked production: at `alpha=0.5, AR=3`, seven of eight paired seeds
showed a shift in rotational kurtosis between `dt=0.01` and `dt=0.005`. The
paired mean was `-0.002525 +/- 0.000796` (standard error; two-sided `p=0.0157`).
Protocol v3 therefore makes `dt=0.005` the uniform production step and compares
it against `dt=0.0025` before releasing the sweep.

## What failed in job 43344190

The copied result directory contains:

- 290 trajectory and moment streams;
- 100 final result JSON files and 100 final histogram files;
- 190 missing final results;
- completed cases only at `alpha=0.95` and `alpha=1`;
- failed inelastic trajectories stopping between about 148 and 1100
  collisions per particle.

The failure was not physical fast cooling by itself. Under the similarity
thermostat, the old code multiplied the persistent NTC majorant by every
reheating factor but never removed the cooling that the reheating undid. Thus
the proposal count inherited a secular factor approximately

\[
  g_{\max}^{\rm old}(\tau) \sim
  \exp[C(1-\alpha^2)\tau],
\]

even though the reduced HCS distribution was stationary. The reusable NTC
arrays then grew until the 4-GiB Slurm cgroup killed the process. Increasing
the memory request would only postpone this algorithmic failure.

The repaired update carries only the maximum relative speed observed in the
current step into rescaled units. It also checks a candidate-count ceiling
before allocating buffers. A full local protocol-v2 realization at
`alpha=0.5, AR=1.35, N=2000, tau=60` measured:

| diagnostic | measured value |
|---|---:|
| runtime (two identical reruns) | 85.18-110.36 s |
| peak RSS | about 684 MiB |
| initial/final NTC majorant ratio | 1.000 |
| mean candidates per step | 7.07 |
| peak candidates per step | 8 |
| configured candidate ceiling | 100,000 |
| majorant violations | 0 |
| same-step repeated-particle pairs | 38/60,000 (0.063%) |
| closure support exits | 0 |

## Why uniform rescaling preserves the HCS physics

For a hard-particle collision law, scale translational and angular velocities
with the same positive factor:

\[
  \mathbf v^*=s\mathbf v,\qquad
  \boldsymbol\omega^*=s\boldsymbol\omega,\qquad
  E_r^*=s^2E_r.
\]

Then `c`, `w`, the temperature ratio, all reduced cumulants, the rotation
number used by the collision-area correction, and every conditional energy
fraction are unchanged. Relative speed, angular speed, and collision frequency
all acquire the same factor. Consequently, rotation accumulated per collision
is also unchanged. The transformed run changes the laboratory-time
parameterization, not the trajectory in collision-count time.

The NTC acceptance probability is invariant only if its speed majorant is in
the same velocity units as the current state. In a stationary rescaled HCS it
must remain statistically stationary; it must not accumulate the product of
all earlier reheating factors. Protocol v3 bounds majorant exceedances per
accepted collision, the thermal-unit majorant, and the pre-allocation candidate
ceiling.

Protocol v3 also counts actual collision pairs that reuse either particle in
one DSMC step. This is a direct time-discretization diagnostic: the engineering
gate requires the repeated-pair fraction to remain below 1%. The paired
`dt=0.0025` arm is the stronger, observable-level convergence test; the reuse
counter explains a failure instead of silently treating the step size as
innocuous.

## Transfer of the rough-particle diagnostics to spherocylinders

The attached paper has `d_t=d_r=3` for rough spheres and uses
`theta=T_rot/T_tr`. An axisymmetric spherocylinder has three translational and
two dynamically active rotational degrees of freedom. This campaign therefore
uses

\[
  a_{20}=\frac{4}{15}\langle c^4\rangle-1,\qquad
  a_{02}=\frac12\langle w^4\rangle-1,\qquad
  a_{11}=\frac23\langle c^2w^2\rangle-1.
\]

The repository closure coordinate is the reciprocal convention,
`T_tr/T_rot`. Protocol v3 writes both ratios explicitly and plots
`T_rot/T_tr` in the paper-style figure, but numerical controls use
`log(T_tr/T_rot)`, whose absolute difference is invariant under taking the
reciprocal. The sphere-only cumulant involving
`(c dot w)^2` as an independent equal-dimensional invariant is not transferred
unchanged. Instead the campaign retains the spherocylinder material-axis
signals

\[
  A_{cu}=\left\langle(\mathbf c\cdot\mathbf u)^2-c^2/3\right\rangle,
\]

and

\[
  A_{cw}=\left\langle(\mathbf c\cdot\mathbf w)^2-c^2w^2/3\right\rangle.
\]

The distribution plot now includes the complete Sonine expressions for
`phi_c`, `phi_w`, and `phi_cw` for `d_t=3, d_r=2`. Histogram confidence bands
resample independent realizations. They no longer treat correlated snapshots
and particles from one realization as independent Poisson observations.

High-velocity fits remain descriptive unless they pass all of the following:
tail-count, occupied-bin, leave-one-realization-out, threshold-stability, and
fit-quality gates. The present weakly non-Gaussian states are expected to
resolve the intermediate range much more reliably than a true asymptotic
exponent.

## Elastic singular point

For every aspect ratio at `alpha=1`, the fitted closure is bypassed. Each
accepted pair collision exchanges translational and rotational energy with
probability

\[
  1/Z_r=3/5,\qquad Z_r=5/3.
\]

On exchange, the pair translational fraction is sampled from `Beta(2,2)` and
the rotational energy is split uniformly between the two particles. This
conserves total pair energy and has equipartition as its invariant state. The
elastic cases are controls, not the target scientific regime.

## Protocol-v3 staged design

### Engineering gate

Five physical cases bracket maximum cooling, shape, the near-elastic regime,
and the exact elastic block:

`(alpha,AR)={(0.5,1.35),(0.5,3),(0.8,2),(0.95,2),(1,3)}`.

Each uses eight paired seeds, the production population `N=10000`, and three
arms: rescaled `dt=0.005`, unscaled `dt=0.005`, and rescaled `dt=0.0025`. The
gate requires complete output, stationarity, zero runtime repairs, no
closure-domain exits, at most `1e-5` majorant exceedances per accepted pair, a
stationary majorant, less than 1% same-step particle reuse, and paired
statistical consistency. The numerical-control
comparison has an explicit three-standard-error resolution ceiling of `0.01`
for cumulants/correlations and `0.02` for temperature ratios. It separately
reports whether the 95% interval establishes the tighter practical-equivalence
floor (`0.002` and `0.01`, respectively); lack of significance alone is not
misreported as proof of equivalence.

### Scientific sweep

The sweep has 37 physical cases and ten independent realizations per case:

- `alpha={0.5,0.6,0.7,0.8,0.9,0.95,1}` at `AR={1.5,2,3}`;
- `AR={1.1,1.2,1.35,2.5}` at calibrated
  `alpha={0.5,0.8,0.95,1}`.

Every task uses `N=10000`, `dt=0.005`, runs to `tau=1500`, and samples `tau=500..1500`
every 5 collisions per particle. Including `AR=1.1` and `1.2` is deliberate:
the slowest artifact-predicted HCS relaxation time is about 108 collisions per
particle, so the first retained sample occurs after more than four such times.

Protocol-v2 timings show that halving the step increases wall time by about
8-23%, not twofold, because candidate-processing work is nearly unchanged.
Budget roughly 4.5 core-hours for the most expensive full task and about 1,650
core-hours for the sweep: approximately 6.5 hours of ideal execution at 256
concurrent cores or 13 hours at 128 cores, before queueing and case variation. The
submission requests eight hours per sweep task and 1.5 GiB per task.

Parallelism is exclusively across independent cases and realizations. Each
realization remains single-core with BLAS/OpenMP thread counts fixed to one;
the stateful collision sequence is not split across threads.

## Negishi submission

Negishi standard CPU nodes have 128 cores and 256 GiB. The measured 684-MiB
peak permits 128 one-core workers per node with a 1.5-GiB request per worker.
The normal CPU queue is used because the sweep can exceed the four-hour
standby limit. Set concurrency to the account allocation actually available.

The angular-evidence artifact is intentionally not tracked by Git. Before
submission, copy both `closure_v2.npz` and its adjacent `manifest.json` to the
same repository-relative directory on Negishi. Verify the NPZ there with

```bash
sha256sum models/microscopic_closure_v2_angular_evidence/closure_v2.npz
```

The required digest is
`f6a6cbec02ea83fa8bee0a5d5e80b575f5f1b8fc046d5f5cbcd098946ee3f616`.
The submission preflight recomputes this digest and checks the adjacent
manifest before any Slurm job is created.

Protocol v7 changes the orientation integrator, so no earlier engineering
result is reusable. Run the 24-task worst-case numerical gate first:

```bash
HCS_NG_MAX_CORES=24 \
bash hpc/submit_hcs_ng_campaign.sh \
  numerics-pilot \
  models/microscopic_closure_v2_angular_evidence/closure_v2.npz \
  numerics_pilot_angular_v7
```

Only after that summary passes, run the new engineering gate:

```bash
HCS_NG_MAX_CORES=120 \
bash hpc/submit_hcs_ng_campaign.sh \
  engineering \
  models/microscopic_closure_v2_angular_evidence/closure_v2.npz \
  engineering_angular_v7 \
  results/hcs_ng_numerics_pilot_angular_v7/summary.json
```

After its QA job reports `study_campaign_pass: true`, run the 80-task
long-time sentinel; do not submit a larger stage until its summary passes:

```bash
ENGINEERING_SUMMARY=results/hcs_ng_engineering_angular_v7/summary.json
HCS_NG_MAX_CORES=80 \
bash hpc/submit_hcs_ng_campaign.sh \
  stability-sentinel \
  models/microscopic_closure_v2_angular_evidence/closure_v2.npz \
  stability_sentinel_angular_v7 \
  "$ENGINEERING_SUMMARY"
```

After `results/hcs_ng_stability_sentinel_angular_v7/summary.json` reports
`long_time_stability_campaign_pass: true`, run the 148-task two-sided
stability gate:

```bash
HCS_NG_MAX_CORES=148 \
bash hpc/submit_hcs_ng_campaign.sh \
  stability \
  models/microscopic_closure_v2_angular_evidence/closure_v2.npz \
  stability_angular_v7 \
  results/hcs_ng_stability_sentinel_angular_v7/summary.json
```

Only after `results/hcs_ng_stability_angular_v7/summary.json` reports
`long_time_stability_campaign_pass: true`, submit the 370-task production
sweep:

```bash
HCS_NG_MAX_CORES=256 \
bash hpc/submit_hcs_ng_campaign.sh \
  sweep \
  models/microscopic_closure_v2_angular_evidence/closure_v2.npz \
  sweep_angular_v7 \
  results/hcs_ng_stability_angular_v7/summary.json
```

Do not use the old `hcs_ng_engineering_baseline_v1`, failed
`hcs_ng_engineering_angular_v2`, or any pre-v7 summary. The
preflight requires complete passing results from the immediately preceding
stage using the same artifact bytes.

## Relevant implementation

- `DSMC_0D_v2/src/dsmc_v2/simulation.py`: similarity majorant and
  pre-allocation NTC ceiling;
- `DSMC_0D_v2/src/dsmc_v2/non_gaussian.py`: explicit temperature-ratio
  conventions and streaming reduced observables;
- `DSMC_0D_v2/scripts/make_hcs_ng_manifest.py`: staged protocol-v7 designs;
- `DSMC_0D_v2/scripts/run_hcs_ng_task.py`: artifact/correction routing and
  provenance;
- `DSMC_0D_v2/scripts/analyze_hcs_ng_campaign.py`: replicate-level statistics,
  Sonine comparisons, tail gates, and separated study/deployment verdicts;
- `hpc/check_hcs_ng_prerequisites.py`: immutable artifact and pilot gate;
- `hpc/submit_hcs_ng_campaign.sh` and `hpc/hcs_ng_array.slurm`: Negishi job
  arrays and resource requests.

## RCAC references

- Negishi overview: https://docs.rcac.purdue.edu/userguides/negishi/overview/
- Running jobs: https://docs.rcac.purdue.edu/userguides/negishi/run_jobs/
- Job submission matrix:
  https://www.rcac.purdue.edu/knowledge/negishi/run/slurm/queues/job-submission-matrix
