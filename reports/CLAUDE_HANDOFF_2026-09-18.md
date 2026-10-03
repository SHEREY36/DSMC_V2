# Closure-v2 / DSMC hand-off — 2026-09-18

> **2026-09-23 HCS-NG addendum:** the protocol-v3 sweep documented in the
> 2026-09-22 audit exposed a delayed apparent energy-partition instability and
> is not a stationary HCS data set. The root cause was subsequently identified
> as similarity reheating of roundoff in the conserved centre-of-mass velocity:
> the old code counted that bulk mode as translational temperature. The active
> replacement computes peculiar temperature and projects to zero momentum
> before every HCS rescale. The subsequent v5 sentinel confirmed that repair
> but exposed NTC initialization, correction-support, statistical-stationarity,
> and `dt=0.005` convergence failures. Protocol v6 repaired those issues, but
> its complete engineering gate resolved an `a20` splitting error at
> `alpha=0.5, AR=1.35`. The active replacement is documented in
> `reports/HCS_NG_PROTOCOL_V7_2026-09-23.md`: it uses a symmetric midpoint
> orientation integrator and adds a 24-task worst-case numerical gate before
> the 120-task engineering gate, then an 80-task long-time sentinel followed by
> a 148-task, two-sided,
> `chi=(1-alpha^2)tau=600` stability gate before the 370-task sweep.
> The 80-task sentinel completed successfully but its original analysis was a
> false negative: 32 runs missed only the terminal scheduled sample, and one
> low-N `a02` comparison duplicated the precision obligation already passed by
> the mandatory high-N engineering gate. Analysis revision
> `hcs-ng-analysis-v2` repairs both without changing measured data or dynamics;
> see `reports/HCS_NG_SENTINEL_V7_DIAGNOSIS_2026-09-25.md`.
> Partial/aborted trajectories are
> diagnostic only, and scientific figures are released only after the complete
> stability and production verdicts pass. USF remains non-gating.

This document is the continuation point for a new agent.  It separates what is
scientifically established from what is only diagnostic, records the local
working tree, and identifies the exact changes still needed before another
large Negishi submission.

## 1. Executive status

The full correction-excitation campaign is complete and should **not** be run
again.  It produced 15,552/15,552 valid excitation observations at all 144
physical baseline nodes, with no execution failures.  The data answer the
release question:

- the fitted angular response is strong in the isolated excitation test;
- the fitted energy response degrades the conditional energy distribution;
- the angular-only response is mostly outside its calibrated feature domain in
  the exploratory inelastic USF cases, but the current DEM stress reference is
  under repair and cannot provide a valid comparative verdict;
- therefore no invariant correction is presently safe to deploy across HCS
  and USF.

The current defensible candidate is the existing full-domain schema-2.4
sampler artifact run with:

```yaml
microscopic_closure:
  routing: variational_v2
  angular: variational_v2
  artifact: models/microscopic_closure_v2_usf_candidate/closure_v2.npz
  invariant_corrections: false
  state_update_cpp: 0.05
```

This selection is encoded in
`DSMC_0D_v2/config/full_domain_baseline_candidate.yaml`.

The model is a **good-enough frozen baseline candidate**, not yet a general
production release.  HCS validation is active.  USF validation is explicitly
deferred until the corrected LAMMPS/DEM reference is frozen; current USF stress
numbers must not be used for model tuning or release.  The next compute is HCS
validation, not another fit/tuning loop.

## 2. Repository and environment state

- Repository root: `/home/muhammed/Documents/Thesis/DSMC_V2`
- Branch: `closure-v2-repairs`
- Starting commit for the uncommitted work: `260a51599a5ab96fb0a122c9536a34ea7fe8ab9b`
- Python wrapper used for all checks: `hpc/python.sh`
- Last verified environment included Python 3.11, NumPy 2.4.6, SciPy 1.17.1,
  matplotlib 3.11.1, scikit-learn 1.9.1, and PyYAML 6.0.3.

The changes described here are intentionally left uncommitted.  Do not use
`git add -A`: the tree contains user-owned and generated files that must not be
folded into a code commit.

Modified tracked files from this work:

- `Coll_Models_v2/scripts/compile_correction_surface.py`
- `Coll_Models_v2/src/coll_models_v2/artifact.py`
- `Coll_Models_v2/src/coll_models_v2/correction_support.py`
- `Coll_Models_v2/tests/test_artifact_inputs.py`
- `Coll_Models_v2/tests/test_response.py`
- `DSMC_0D_v2/scripts/analyze_usf_validation.py`
- `DSMC_0D_v2/scripts/make_hcs_validation_manifest.py`
- `DSMC_0D_v2/scripts/make_usf_validation_manifest.py`
- `DSMC_0D_v2/scripts/plot_hcs_validation.py`
- `DSMC_0D_v2/scripts/run_hcs_validation_task.py`
- `DSMC_0D_v2/src/dsmc_v2/simulation.py`
- `DSMC_0D_v2/src/dsmc_v2/state.py`
- `DSMC_0D_v2/tests/test_state.py`
- `hpc/hcs_validation_array.slurm`
- `hpc/submit_hcs_validation.sh`
- `tests/integration/test_hpc_stages.py`

New source/config/report files from this work:

- `Coll_Models_v2/scripts/build_selective_evidence_artifact.py`
- `Coll_Models_v2/scripts/plot_full_candidate_validation.py`
- `DSMC_0D_v2/config/full_domain_baseline_candidate.yaml`
- `DSMC_0D_v2/scripts/plot_local_model_evidence.py`
- `hpc/submit_hcs_learned_low.sh`
- `reports/model_readiness_local_evidence_2026-09-18.md`
- `reports/CLAUDE_HANDOFF_2026-09-18.md`

Known pre-existing user-owned untracked paths; preserve them and do not assume
ownership:

- `Coll_Models_v2/scripts/export_artifact_parameter_data.py`
- `models/microscopic_closure_v2_candidate/`
- `models/microscopic_closure_v2_usf_candidate/`
- `output/`, `outputs/`, and `tmp/`
- `reports/closure_v2_parameters_and_excitation_physics.md`

Generated evidence that normally should not be committed:

- `models/microscopic_closure_v2_angular_evidence/` (about 444 MB)
- the full local result directories listed below
- PNGs under `reports/figures/` unless the user explicitly wants figures under
  version control

Before committing, inspect with `git status --short`, stage only named source,
test, config, and report files, run `git diff --cached --check`, and review the
cached file list.

## 3. Completed data and provenance

### Baseline surface

`manifests/artifact_grid.csv` has 144 baseline rows.  The physical coordinate
axes represented in the artifact are:

- restitution coefficient: `alpha = {0.5, 0.8, 0.95, 1.0}`;
- aspect ratio: `AR = {1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0}`;
- temperature-ratio coordinate values appearing in the sparse surface:
  `theta = {0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 1.0, 2.0}`.

The 144 points are a sparse physical surface, not a Cartesian product of all
three lists.  Runtime interpolation uses complete-cell multilinear weights
where possible and barycentric interpolation on the measured convex hull for
incomplete cells.  It fails closed outside that hull.

### Completed excitation surface

- Manifest: `manifests/excitation_full_candidate_v1_combined.csv`
- Rows: 15,552
- Physical nodes: 144
- Full-rank designs: 144/144
- Execution failures: 0
- Offline validation:
  `results/closure_estimates/excitation_full_candidate_v1_combined_offline_validation.json`

Important global held-out relative RMSE values:

| parameter | relative RMSE |
|---|---:|
| eta1 | 0.01964 |
| eta2 | 0.02237 |
| lambda1 | 0.01654 |
| lambda2 | 0.01506 |
| lambda3 | 0.02221 |
| lambda4 | 0.01213 |

Distribution-level evidence is the deciding result:

| diagnostic | baseline p95 | fitted/deployed p95 | verdict |
|---|---:|---:|---|
| held-out angular quantile W1 | 0.03572 | 0.00846 | improves |
| central angular quantile W1 | 0.01801 | 0.00224 | improves |
| held-out energy quantile W1 | 0.01783 | 0.03001 tangent | worsens |
| held-out energy quantile W1 | 0.01783 | 0.02461 exact fit | worsens |
| central energy quantile W1 | 0.00759 | 0.00869 tangent | worsens |

Consequently `distribution_response_pass=false` and the production correction
artifact was correctly blocked.  The Slurm artifact pack job was dependent on
that scientific QA with `afterok`, which is why the HPC chain produced no new
NPZ.  The missing NPZ was deliberate fail-closed behavior, not lost compute.

## 4. Selective correction repair and evidence artifact

The local repair introduces release policy
`validated-angular-only-v1`.  It independently evaluates `eta1` and `eta2`,
zeros all energy-response rows (`lambda1` through `lambda4`), suppresses
nonlinear angular rows, and suppresses a complete node if its central sentinel
failed.

Compiled coefficient evidence:

- file:
  `results/closure_estimates/excitation_full_candidate_v1_angular_evidence_coefficients.json`
- coefficient nodes: 144
- both eta rows released: 122 nodes
- eta1 only released: 19 nodes
- angular release of any kind: 141 nodes
- fully suppressed: 3 nodes
- energy release: 0 nodes
- trust amplitude: 0.5 at 141 nodes and 0.25 at the 3 suppressed nodes

The three fully suppressed physical coordinates `(alpha, theta, AR)` are:

1. `(0.5, 0.025, 1.1)`
2. `(1.0, 0.025, 1.1)`
3. `(1.0, 0.1, 1.35)`

Evidence artifact:

- NPZ: `models/microscopic_closure_v2_angular_evidence/closure_v2.npz`
- manifest: `models/microscopic_closure_v2_angular_evidence/manifest.json`
- SHA-256:
  `f6a6cbec02ea83fa8bee0a5d5e80b575f5f1b8fc046d5f5cbcd098946ee3f616`
- size: about 444 MB
- schema: 2.4.0
- status: `evidence_only_not_deployable`
- runtime load check: passed
- stability check: passed
- maximum energy interpolation error: `1.9998732626e-4`

`build_selective_evidence_artifact.py` preserves the already validated base
sampler arrays and replaces only the beta correction tensors.  It does not
rebuild the large quantile tables.  `artifact.py` also rejects a non-strict
release policy in the normal production builder, preventing this evidence
policy from silently becoming a deployment artifact.

The approximately 444 MB size is expected for schema 2.4: it preserves the
adaptive energy quantile tables and lambda2/lambda3 energy-shape sensitivity
tables.  The old approximately 57 MB schema-2.3 artifact omitted those
sensitivities.  The large size is not evidence of lower accuracy, and forcing
the schema-2.4 artifact down to the old size would remove required information.

## 5. Independent fresh-CTC result

The existing full-domain candidate artifact is:

- `models/microscopic_closure_v2_usf_candidate/closure_v2.npz`
- SHA-256:
  `14c8ce9156b077cf62d314d20fd4fe4678582c1aa6bc03c4154f47aea209fba6`
- size: about 444 MB

Its independent fresh-CTC validation is recorded in
`results/closure_estimates/independent_ctc_holdout_usf_candidate_v1_validation.json`:

- expected/valid/failures: 36/36/0
- validation pass: true
- response pass: true
- minimum effective-sample-size fraction: 0.60355
- maximum importance weight share: 1.361e-4

This holdout validates the response calculation on those exact artifact bytes.
For the selected baseline configuration the corrections are disabled, so it
should be retained as provenance/integrity evidence but must not be presented
as proof that corrections should be deployed.

## 6. Local HCS evidence

The reduced HCS design uses the six independent DEM-backed or exact-elastic
nodes
`alpha={0.8,0.95,1.0} x AR={2,3}`, two initial temperature partitions and two
realizations.  These are diagnostics, not the final high-statistics gate.

Angular evidence artifact, corrections enabled, `tau=12`:

- all 6/6 cases pass the reduced physics criteria;
- closure overhead is about 3.2--5.3%;
- top-level production is deliberately false because this is evidence mode.

Frozen sampler, corrections disabled, `tau=12`:

- 5/6 cases pass;
- `(alpha,AR)=(0.8,3)` misses only the drift threshold:
  `0.10183` versus the `0.10` cutoff.

Frozen sampler, corrections disabled, extended to `tau=20` for `(0.8,3)`:

- physics pass: true;
- mean theta: 0.97732 versus DEM target 1.0158;
- initial-condition spread: 0.01361;
- maximum replicate-mean relative drift: 0.09227;
- closure overhead: 2.75%.

Interpretation: the baseline sampler is adequate at all six truth-backed HCS
nodes after allowing the transient to decay.  This result does not prove all
28 artifact alpha/AR axis pairs; that is the next full-domain HCS task.

### Learned low-theta near-sphere diagnostic

A new correction-disabled design tested the region the earlier gate missed:
`alpha={0.5,0.8,0.95,1.0}`, `AR={1.1,1.2,1.35}`, and exact learned starts
`theta0={0.0125,0.2}`.  Two 1,000-particle replicas per start were integrated
to `tau=30`.  Exact initial modal-temperature normalization removes the
finite-sample displacement outside a boundary node; uniform HCS similarity
rescaling prevents numerical freezing while preserving theta versus collision
count.

All 48 low-side runs have zero domain exits and zero energy repairs/clamps.
Seven of twelve cases pass at `tau=30`.  The five remaining cases are still
relaxing: all four `AR=1.1` cases and `(alpha,AR)=(0.5,1.2)`.  This is not a
model failure.  The full table is in
`reports/model_readiness_local_evidence_2026-09-18.md`; the summary is
`results/hcs_learned_low_local/summary.json`, and the figure is
`reports/figures/hcs_learned_low_local.png`.

The dedicated follow-up `hpc/submit_hcs_learned_low.sh` runs all twelve cases
uniformly with 2,000 particles, three replicas, and `tau=120` (72 one-core
tasks).  It passes `false true true` to the HCS worker: corrections disabled,
exact initial modal temperatures enabled, HCS similarity rescaling enabled.

One separate hot-boundary run at `(alpha,AR,theta0)=(1,1.1,2)` left the
physical hull after a stochastic collision moved theta infinitesimally above
the maximum learned coordinate.  Do not clamp or extrapolate this.  Robust use
of the exact upper boundary requires support beyond theta=2.

Relevant outputs:

- `results/hcs_angular_evidence_local/summary.json`
- `results/hcs_baseline_evidence_local/summary.json`
- `results/hcs_baseline_extended_local/summary.json`

## 7. Local USF diagnostics — not a validation gate

The user has identified errors in the current LAMMPS/DEM USF stress tensor and
is repairing the reference.  Temperature may be correct, but it is not yet a
frozen independent reference.  Preserve the following runs as diagnostics;
do not use their RMSE values to accept, reject, or tune the closure.

The paired reduced comparison used `(alpha,AR)={0.8,0.95} x {2,3}`, two
realizations per arm, 1,000 particles, and `tau=30`.

| arm | stress relative RMSE | theta relative RMSE |
|---|---:|---:|
| corrections disabled | 0.03583 | 0.01892 |
| angular-only correction | 0.03711 | 0.03332 |

Angular-only correction out-of-domain fractions were approximately:

- 97% at `(0.8,2)`;
- 92% at `(0.8,3)`;
- 25% at `(0.95,2)`;
- 0% at `(0.95,3)`.

The out-of-domain measurements are internal model-support diagnostics and
remain meaningful, but the RMSE comparison is not a trustworthy cross-flow
accuracy verdict.  Keep invariant corrections disabled until both feature
support and the repaired reference have been validated.

Uncorrected extension results:

- `(0.8,2)` at `tau=80`: stationary; provisional-reference stress and theta
  errors were 5.51% and 4.62%;
- `(0.95,2)` at `tau=80`: still warming;
- `(0.95,2)` at `tau=200`: stationary, theta drift 0.00642 and
  total-temperature drift 0.06611; provisional-reference stress and theta
  errors were 2.21% and 2.23%.

The near-elastic stationarity failure at shorter time was therefore an
insufficient-run-length issue, not a model-fit failure.  Accuracy remains
unresolved until the repaired USF reference exists.

Relevant outputs:

- `results/usf_angular_evidence_local/summary.json`
- `results/usf_baseline_extended_local/summary.json`
- `results/usf_baseline_near_elastic_local/summary.json`

The local summaries intentionally keep their top-level deployment verdicts
false because they are small evidence designs.  Inspect their per-arm and
per-case fields for the scientific results above.

## 8. Model capability that may be claimed now

The physical interpolation surface is calibrated for `0.5 <= alpha <= 1.0`
and `1.1 <= AR <= 3.0`, subject to the sparse three-dimensional physical hull
in `(alpha, theta, AR)`.  Alpha and AR are interpolable inside that hull; they
are not merely labels for independent models.  Runtime queries outside the
hull fail closed.

Do **not** claim `AR=1.0` from this artifact.  The sphere endpoint is absent.
If the promised domain must be `1 <= AR <= 3`, implement and validate an exact
sphere branch at `AR=1`; do not extrapolate from `AR=1.1`.

The design goal remains one closure across flows.  The fact that selected
validation nodes are labeled HCS or USF does not create flow-specific model
ownership.  The current evidence instead says that the invariant correction
law is not yet flow-transferable, hence corrections are globally disabled.

Claims that are justified:

- the 144-node sampler surface and interpolation mechanism are built and load;
- fresh-CTC response validation passed on the base candidate bytes;
- reduced truth-backed HCS checks pass with the frozen sampler;
- exploratory USF temperature dynamics become stationary after sufficiently
  long collision time, but USF accuracy is not currently validated;
- the current corrections must remain off.

Claims that are not yet justified:

- full-domain HCS release;
- full-domain USF release;
- production readiness;
- `AR=1` coverage;
- paper-scale non-Gaussian statistics.

## 9. Code changes completed locally

### Correction release and artifact safety

- `correction_support.py`: independent held-out angular support assessment.
- `artifact.py`: strict versus `validated-angular-only-v1` release policies;
  selective zeroing/suppression; explicit rejection of evidence policies by
  the ordinary cached production builder.
- `compile_correction_surface.py`: command-line release-policy selection.
- `build_selective_evidence_artifact.py`: streaming evidence artifact builder
  that preserves the base arrays and validates coordinate coverage/loadability.

### Validation and plotting

- `make_hcs_validation_manifest.py`: evidence mode and repeatable `--case`
  filters.
- `make_usf_validation_manifest.py`: evidence mode, replicate count, alpha,
  AR, and arm filters.
- `analyze_usf_validation.py`: handles an uncorrected-only arm without a
  `KeyError`; corrected/uncorrected plot semantics repaired.
- `plot_hcs_validation.py`: evidence-only verdict handling.
- `plot_full_candidate_validation.py`: correction-campaign validation figure.
- `plot_local_model_evidence.py`: combined HCS/USF local evidence figure.

### Tests

The complete test command passed:

```bash
PYTHON=hpc/python.sh MPLCONFIGDIR=/tmp/matplotlib-dsmc make test
```

Results:

- integration: 43 passed;
- `Coll_Models_v2`: 155 passed;
- `DSMC_0D_v2`: 35 passed;
- total: 233 passed.

`git diff --check` also passes.

## 10. Figures ready for inspection/presentation

- `reports/figures/full_candidate_excitation_validation.png`
- `reports/figures/hcs_angular_evidence_local.png`
- `reports/figures/hcs_baseline_evidence_local.png`
- `reports/figures/hcs_baseline_extended_local.png`
- `reports/figures/hcs_learned_low_local.png`
- `reports/figures/usf_angular_evidence_local.png`
- `reports/figures/usf_baseline_extended_local.png`
- `reports/figures/usf_baseline_near_elastic_local.png`
- `reports/figures/local_model_evidence_comparison.png`

The first and final figures are the best concise presentation pair: one shows
why energy corrections are rejected despite good angular response, and the
other shows HCS/USF consequences of the release decision.

Plots can be regenerated with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-dsmc hpc/python.sh \
  Coll_Models_v2/scripts/plot_full_candidate_validation.py

MPLCONFIGDIR=/tmp/matplotlib-dsmc hpc/python.sh \
  DSMC_0D_v2/scripts/plot_local_model_evidence.py
```

## 11. Critical HPC workflow warning

Do not submit `hpc/submit_full_model_pipeline.sh` again.  It is the completed
correction-building pipeline; rerunning it would purchase thousands of
unnecessary fits and again block on the correction QA that has already answered
the scientific question.

The existing release submission scripts are not yet aligned with the new
baseline decision:

- `make_usf_validation_manifest.py --mode full` currently creates only the
  `corrected` arm;
- `submit_usf_validation.sh` requires the old corrected-HCS and response-pilot
  chain;
- the integrated full-model pipeline passes `true` to HCS correction tasks;
- `run_hcs_ng_task.py` currently forces `invariant_corrections=True` for every
  nonspherical non-Gaussian run;
- `check_hcs_ng_prerequisites.py` checks `beta_coordinates`, which is a
  correction-surface prerequisite, rather than the baseline sampler's
  `surface_coordinates`.

Those are workflow semantics, not evidence that the base model is wrong.  A
successor should first implement a separate baseline-release path.  Do not
weaken existing corrected gates in place; retain them as historical machinery
and make the baseline intent explicit.

## 12. Required next work, in order

### A. Commit the finished local repair safely

1. Review this report and the shorter readiness report.
2. Inspect every diff.
3. Stage only the named source/test/config/report files.
4. Exclude the 444 MB NPZ, bulk results, `tmp/`, and user-owned untracked files.
5. Run the full 233-test suite and `git diff --cached --check` once more.
6. Commit and push `closure-v2-repairs` only after the staged list is clean.

### B. Add an explicit HCS baseline release workflow

Implement and test a new path with these non-negotiable semantics:

1. Artifact is fixed to SHA
   `14c8ce9156b077cf62d314d20fd4fe4678582c1aa6bc03c4154f47aea209fba6`.
2. `invariant_corrections=false` for every nonspherical HCS task; retain the
   same rule for future USF work after its reference is repaired.
3. No fitting, coefficient compilation, artifact rebuild, or excitation jobs.
4. HCS full-domain manifest is generated from `surface_coordinates` and covers
   all 28 alpha/AR axis pairs, two initial theta values, and at least three
   replicates: 168 independent tasks.
5. Exact-versus-cached state-update cadence remains checked so speed does not
   alter the physics.
6. A learned-extremes design starts at exact far-from-equipartition artifact
   nodes, normalizes finite ensembles to the requested modal temperatures, and
   uses uniform HCS similarity rescaling to avoid numerical freezing without
   changing collision-count dynamics.
7. Use one core per independent realization and parallelize across cases and
   replicates.  Do not reduce particle count, collision count, sample cadence,
   or replicate count to gain speed.
8. Preserve fail-closed artifact-hash checks and fresh output directories.

Recommended implementation shape:

- add `submit_baseline_release_validation.sh` rather than overloading the
  correction-build pipeline;
- pass `false` explicitly to `hcs_validation_array.slurm`;
- add integration tests asserting task counts, artifact hashing, arm choice,
  correction flags, and Slurm dependency direction.

A future correction-disabled USF workflow is still needed, but it must not be
run or interpreted before the repaired reference is frozen.

### C. Run full-domain frozen validation on Negishi

Only after B is reviewed and tested:

1. run the learned-extremes near-sphere HCS validation;
2. run full-domain HCS reference/cached-cadence validation;
3. inspect physical, stationarity, precision, performance, and hash verdicts;
4. do not run the 280-task USF accuracy grid until the LAMMPS/DEM reference is
   repaired and frozen;
5. inspect per-case results, not only aggregate metrics;
6. do not refit in response to an isolated stochastic failure—first distinguish
   incomplete stationarity, sampling uncertainty, domain exit, and true model
   bias.

The next campaign should be hundreds of jobs, not another 15,000-job
excitation campaign.

### D. Promote only if both flows pass

If the same frozen bytes pass full-domain HCS, they may be released for a
clearly labeled HCS-only non-Gaussian campaign.  General cross-flow promotion
still requires the future repaired-reference USF validation.  At promotion:

1. write a release manifest containing the exact artifact SHA, code commit,
   validation summary hashes, supported domain, and correction-off decision;
2. promote/copy the candidate to the production model location;
3. update default production configuration only then;
4. keep the evidence artifact separately labeled and nondeployable.

If a full-domain case fails, classify it before changing the model.  A failure
caused by insufficient collision time or Monte Carlo precision calls for a
validation extension, not a new excitation fit.  A repeatable physical bias or
sampler-hull failure requires a scoped model decision and new evidence.

### E. Adapt and then launch the non-Gaussian HCS work

The non-Gaussian infrastructure is already tracked in commit history:

- streaming observables in `DSMC_0D_v2/src/dsmc_v2/non_gaussian.py`;
- manifest, runner, and analyzer scripts;
- Slurm array and staged submitter;
- integration tests.

It measures `theta`, `a20`, `a02`, `a11`, `A_cu`, and
`A_cw_quadrupolar`, estimates integrated autocorrelation time, uses
realization-level bootstrap intervals, records logarithmic histograms, removes
the correct radial Jacobian before tail fitting, and refuses tail fits below a
declared pooled tail count.

Before scientific use, adapt the runner and prerequisite check to the approved
baseline candidate:

- use `invariant_corrections=false` for nonspherical cases;
- validate `surface_coordinates`, not the correction beta hull;
- require the passing full-domain HCS summary for the same artifact SHA;
- retain scaled/unscaled equivalence in the short engineering pilot;
- keep collision-count sampling and temperature rescaling so late HCS samples
  do not freeze numerically;
- do not start the map/tails production modes before the short pilot establishes
  stationarity, autocorrelation time, and tail count sufficiency.

Existing staged sizes are:

- engineering: 3 cases x 2 scaling arms x 4 replicas = 24 tasks;
- domain pilot: 28 cases x 4 replicas = 112 tasks;
- map: 70 cases x 16 replicas = 1,120 realizations;
- tails: 9 cases x 100 replicas = 900 realizations;
- sphere controls: 4 cases x 16 replicas = 64 realizations.

The Slurm script strides virtual rows across at most the available worker
count, so array-size limits do not require changing the physics.  These modes
should remain staged; do not submit map and tails together before pilot QA.

## 13. Acceptance criteria for “fully ready”

All of the following must be true for the same immutable artifact bytes and
code commit:

- artifact input/grid validation passes;
- runtime load/stability/energy interpolation checks pass;
- independent CTC provenance remains consistent;
- full-domain HCS physical and cadence gates pass;
- for a general HCS+USF deployment, full-domain USF physical, stationary,
  precision, accuracy, and performance gates pass against the repaired frozen
  reference;
- corrections are demonstrably disabled in every release task;
- no runtime repair, clamp, or out-of-domain gate is violated;
- supported domain is stated honestly as `1.1 <= AR <= 3.0` unless an exact
  sphere route is separately validated;
- release record pins hashes and configuration.

Only after all conditions should the artifact be called generally
deployment-ready.  A passing full-domain HCS gate is sufficient for an
explicitly scoped HCS-only non-Gaussian campaign while USF remains deferred.

## 14. Immediate commands for the successor

Read-only orientation:

```bash
cd /home/muhammed/Documents/Thesis/DSMC_V2
git status --short
git branch --show-current
git rev-parse HEAD
sed -n '1,260p' reports/model_readiness_local_evidence_2026-09-18.md
sed -n '1,360p' reports/CLAUDE_HANDOFF_2026-09-18.md
sha256sum models/microscopic_closure_v2_usf_candidate/closure_v2.npz
```

Verification before any commit:

```bash
PYTHON=hpc/python.sh MPLCONFIGDIR=/tmp/matplotlib-dsmc make test
git diff --check
```

The next code action is B above.  There is no scientifically justified reason
to submit another excitation fit, rebuild the sampler, or enable the angular
evidence correction while doing that work.
