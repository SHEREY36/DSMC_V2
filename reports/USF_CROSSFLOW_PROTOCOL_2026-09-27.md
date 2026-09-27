# USF cross-flow validation protocol (2026-09-27)

## Current scientific status

The frozen angular-evidence artifact is HCS-ready, not yet USF-ready.  Its
bytes have SHA-256
`f6a6cbec02ea83fa8bee0a5d5e80b575f5f1b8fc046d5f5cbcd098946ee3f616`.
The complete 360-realization HCS non-Gaussian v8 sweep passed with those exact
bytes.  Energy-response corrections remain suppressed; 141 independently
supported angular-response nodes are released and the runtime falls back to
the base law outside their support.

The purpose of this campaign is therefore validation, not another fit.  DEM is
never used as a calibration target.  Its steady values provide initial guesses
that shorten the transient and an independent qualitative comparison after the
DSMC result is measured.  The exact-sphere Boltzmann-DSMC table is the external
USF accuracy control.

For unthermostatted steady USF the campaign domain is

- `1 <= AR <= 3` and `0.5 <= alpha <= 0.95`;
- `alpha = 1` remains a valid HCS control, but is deliberately excluded from
  steady USF because shear heating has no inelastic sink and no finite steady
  state exists.

The primary rod stress observable is kinetic stress.  The present scalar-v1
rod collision interface does not retain the physical branch vector, so its
collisional/total stress is clearly labeled diagnostic and cannot pass or fail
the model.  Sphere impulse-normal collisional stress remains exact.

## Repairs completed before the campaign

1. An explicit particle count removes the old volume/ceiling ambiguity.
2. Exact finite-ensemble temperature normalization now covers spheres as well
   as rods.
3. Exact-sphere tasks bypass the rod closure and artifact hull completely.
4. USF writes a six-column energy ledger that separately accumulates exact
   shear-map work, accepted-collision energy change, and their residual.
5. Every realization records the artifact, artifact-manifest, and frozen
   reference-provenance hashes.
6. Rod total stress is prevented from silently entering an accuracy gate.
7. The DEM and sphere comparison tables are frozen in the DSMC repository with
   source-repository revision and checksums.

## First-attempt evidence and support-policy repair

The expensive numerics did not crash: all 108 expected realizations completed,
all output schemas and hashes were valid, and every time-step, shear-rate,
sphere, stationarity, branch-convergence, energy-ledger, and performance check
passed. There were no negative-energy repairs or clamps, maximum closure
overhead was 1.825%, and the largest ledger relative residual was below
`1e-13`.

The sole failed condition was the original HCS-style strict correction-support
gate. Corrected rod trajectories at `alpha=0.5` and `0.8` often leave the HCS
excitation box, principally along `PiPi` and occasionally `a2_tr`; the
`alpha=0.95` cases remained supported. This is a cross-flow support result,
not a numerical failure and not evidence for clipping or extrapolating the
learned surface.

The repaired protocol therefore validates an explicit adaptive effective
model. Inside the independently validated response domain it uses the frozen
angular correction. Outside it, the existing fail-closed path suppresses all
response increments and uses the unchanged base collision law. Strict
correction coverage is retained as a separate reported verdict and may remain
false. All other safety, convergence, conservation, stationarity, precision,
and sphere-accuracy gates remain mandatory. The default runtime policy and
the HCS release contract remain strict.

## One-command staged design

The submission creates all manifests and validates every prerequisite before
calling `sbatch`.

| Stage | Realizations | Purpose |
|---|---:|---|
| numerics | 108 | `dt=0.01` versus `0.005`, shear-rate similarity, three sphere controls, and three representative rod aspect ratios |
| pilot | 168 | corrected/uncorrected paired DSMC at `(AR,alpha)={1.5,2,3} x {0.5,0.8,0.95}`, plus sphere controls |
| full | 736 | ten alphas, all seven artifact AR nodes, exact spheres, and 96 off-grid interpolation-holdout runs |

All production measurements use 10,000 particles at `phi=0.01`.  Each physical
coordinate is approached from independent hot and cold initial branches.  The
numerics stage uses two realizations per branch; pilot and full stages use four.
The terminal collision counts are 160, 240, and 360 collisions per particle as
`alpha` approaches 0.95, and only the final 40% is evaluated.

The scientific dependency chain is

```text
numerics -> analyze -> gate
                      |
                      v
pilot    -> analyze -> gate
                      |
                      v
full     -> analyze -> gate -> immutable-byte promotion
```

An array may fail partially and its `afterany` analysis will still report every
missing/crashed task.  The gate then fails closed and prevents the next stage.
Consequently a coding error, truncated task, wrong artifact, mixed provenance,
failed sphere benchmark, lack of stationarity, hot/cold disagreement, excess
sampling uncertainty, unsafe support handling, or energy-accounting error
cannot release the expensive full design.

DEM differences are reported but never enter `stage_pass`.  Rod gates use:

- finite collision-target completion with no energy repairs or clamps;
- correction fallback below 1% is reported as strict correction coverage;
- otherwise, the correction must fail closed to the unchanged base law with
  no other runtime-gate failure;
- energy-ledger closure and late shear/collision power balance;
- hot/cold branch convergence;
- replicate precision and terminal drift;
- time-step convergence and shear-rate similarity in the numerics stage;
- explicit off-grid interpolation cases in the full stage.

Sphere controls additionally gate against the independent Boltzmann-DSMC
temperature and kinetic-stress table.

## Negishi resource mapping

Negishi CPU nodes have 128 AMD EPYC 7763 cores and 256 GB memory.  The
`morri353` allocation exposes 256 purchased cores, so the efficient mapping is
one independent single-threaded realization per core, up to 256 simultaneous
tasks: two complete CPU nodes.  Requesting two nodes for one non-MPI Python
process would not accelerate it.

Measured HCS-v8 peak memory was about 0.64 GiB per task.  The USF array requests
1.8 GiB per core, or 230.4 GiB for 128 tasks on a 256-GB node.  BLAS/OpenMP
thread counts are pinned to one.  The default wall limit is eight hours; it is
an upper bound, not a target runtime.  The three stages contain 1,012 total
realizations but, with 256-way concurrency, only one numerical wave, one pilot
wave, and roughly three full-stage waves are needed if all gates pass.

This conforms to Purdue's documented Negishi hardware and scheduler model:

- <https://rcac.purdue.edu/index.php/compute/negishi>
- <https://docs.rcac.purdue.edu/userguides/negishi/run_jobs/job_submission_matrix/>
- <https://www.rcac.purdue.edu/knowledge/negishi/run/slurm/queues?all=true>
- <https://rcac.purdue.edu/knowledge/negishi/run/slurm/submit>

The normal QOS is used because it can consume purchased cores and permits the
requested walltime; Negishi standby jobs are limited to four hours.  Heavy work
runs only through Slurm, never on a login node.

## Automatic promotion and interpretation

If and only if the full 736-run summary passes, a dependent job copies the
unchanged NPZ bytes to
`models/microscopic_closure_v2_usf_validated_<tag>/` and writes a new manifest
that binds both HCS-v8 and full-USF summaries.  The promoted scope is explicit:

- HCS: `0.5 <= alpha <= 1.0`, `1 <= AR <= 3`;
- steady unthermostatted USF: `0.5 <= alpha <= 0.95`, `1 <= AR <= 3`;
- validated USF observables: reduced translational temperature, temperature
  ratio, kinetic stress, nematic order, stationarity, and energy balance;
- rod collisional/total stress remains outside the validated claim.

No closure coefficient changes during this campaign.  A full-stage failure is
evidence about model support or DSMC dynamics and must be diagnosed honestly;
it is not permission to tune the closure to DEM.
