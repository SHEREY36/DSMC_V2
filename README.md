# DSMC_V2 — BL-compatible variational closure

This repository contains the collision-trajectory generator (`HS_CTC_v2`),
closure estimation and artifact builder (`Coll_Models_v2`), the 0D DSMC
consumer (`DSMC_0D_v2`), and their shared binary/invariant contracts.

Schema 2.2 replaces the experimental schema-2.1 GMM/routing/VSS composition
with an opt-in variational energy-partition and angular kernel. The complete
legacy path remains available for controlled A/B comparisons.

## Frozen physical assumptions

The following are deliberate model choices and are not refitted:

- the v1 no-time-counter candidate clock;
- the v1 polynomial collision cross-section;
- the v1 scalar Borgnakke–Larsen loss draw;
- the v1 CTC normal damping based on centre translational relative velocity;
- the v1 contact integration at 50 steps per Hertzian contact time. Refining it
  is now cheap (`CTC_DT_DIVISOR`), and it carries a measured 1.7% per-event and
  0.5% ensemble-mean discretisation error in the dissipated energy, so this is
  a frozen choice rather than a converged one.

The last restriction is explicitly visible in
`HS_CTC_v2/model/calc_force_dem.f90`. General rigid-body restitution normally
uses contact-point relative velocity, including rotational contributions; the
present thesis model intentionally does not. See the
[primary impact formulation](https://doi.org/10.1016/j.cnsns.2026.109646).

The CTC target transferred to DSMC is the surviving translational energy
partition `z`, not absolute CTC modal production. The frozen BL law remains
authoritative for total loss. No result should claim that the CTC and BL total
losses agree.

## Corrected closure

For every incoming proposal the geometric projected excluded area is

```text
A_perp = pi D^2
       + 2 D L (|u1 x ghat| + |u2 x ghat|)
       + L^2 |ghat dot (u1 x u2)|,
L = (AR - 1) D.
```

`A_perp` is the shadow of a *frozen* pair, and it is not the generator's
acceptance probability. The rods turn while they close, so the acceptance is
the dynamic excluded area, which exceeds `A_perp` by 9 to 33 percent at aspect
ratios 2 and 3 and grows as theta falls. That is physics rather than a staging
artefact: contact needs a centre separation below `L + D`, the generator starts
beyond that, and free rotation preserves an isotropic director law, so the
result cannot depend on the staging distance -- and measurably does not.

The acceptance is nevertheless pure kinematics, since no force acts before
contact. `kinematic_propensity` integrates the force-free encounter over the
impact-parameter plane directly from the stored pre-collision state, and
accepted outcomes are weighted by `1/propensity`, the exact Radon-Nikodym
derivative onto the DSMC's orientation-blind collision measure. Measured
against the generator, the predicted acceptance is within 0.5 percent where
`A_perp` is out by 5 to 25 percent, and with zero spin the integrator
reproduces the analytic `A_perp` to 0.03 percent.

The static weight is retained for A/B runs as `propensity_offsets=None`. The
frozen DSMC cross-section polynomial is reported separately as a clock audit
and is never required to equal a geometric area; note that the true dynamic
cross-section varies by 5 percent at AR = 2 and 23 percent at AR = 3 across
theta in [0.2, 2], which a theta-blind polynomial cannot carry.

The energy kernel is the I-projection of the memoryless Borgnakke-Larsen draw
onto the measured transfer moments:

```text
p(z' | z, eps) proportional to Beta(2,2)(z')
    * exp(lambda1 z' + lambda2 z'^2 + lambda3 z z' + lambda4 eps z').
```

The earlier gated form, `(1 - p_exch) delta(z' - z) + p_exch R(z')`, is
retired. The stored events have no atom at `z' = z`: only 3 to 13 percent of
collisions leave the partition unchanged to 0.01, where the gate needs 25 to 95
percent. Forcing one imposes a floor `p(1-p)(z - mu)^2` on the conditional
variance that the data fall below, which is what produced the negative reset
variances and the `reset_mean = intercept / p_exch` blow-up at AR = 1.1.

Adding `z z'` and `eps z'` to the sufficient statistics keeps the same
I-projection theorem, removes the atom, and makes the dual strictly convex on
sample moments, so the infeasible-moment branch cannot occur. The rotational
collision number survives as the derived lag-one slope of the mean map, and
`lambda4` lets the runtime evaluate the kernel at its own frozen BL loss
instead of inheriting the CTC's. State corrections still act in the natural
parameter, `lambda1 = lambda1_0 + beta dot X`.

`reset_mean` and `reset_second_moment` keep their names in the node estimates
and the artifact, but now report the first two moments of the kernel's
*invariant* law: the partition the kernel drives towards. That is what the
reset law was a proxy for, it coincides with it exactly when the data really
are gated-Beta, and it makes the elastic gate a direct test that an elastic
kernel reaches equipartition.

The angular kernel is

```text
p(cos chi) proportional to exp[eta1 cos chi + eta2 P2(cos chi)],
```

with uniform azimuth about the incoming relative velocity. A `z cos chi`
natural parameter is exported only when the correlation magnitude, confidence
interval, and held-out score gates all pass. Elastic rod angular moments are
allowed to be nonzero.

## Fourteen production invariants

The shared deployed order is:

```text
a2_tr, a2_rot, a11, A_cu,
PiPi, QQ, RtRt, PiQ, PiRt, QRt,
qtr2, qrot2, qtr_qrot, W2
```

`Rt = R + Q` is the irreducible in-plane spin anisotropy. `Acw2` and `vx2`
are retained as diagnostics and never enter the production dot product.
Quadratic cell features use finite-population U-statistics. Tangent spin
directions always satisfy `omega dot u = 0`.

## Data and artifact contracts

Schema `2.2.0` keeps the attempt and outcome records at 200 and 552 bytes.
The former reserved `int32` header is now `ensemble_id`. `load_run()` accepts
completed schema-2.1 shards through a read-only adapter, interpreting their
reserved zero as baseline `ensemble_id=0`; no old shard is rewritten.

The production artifact is `closure_v2.npz`. It contains:

- physical-node coordinates and fitted `p_exch`, energy, and angular surfaces;
- bootstrap uncertainties and optional joint-coupling masks;
- the 14-mode coefficient surfaces, standard errors, and deployment masks;
- feature/diagnostic bounds and event/ESS provenance;
- numerical quantile tables for energy and angle sampling;
- frozen loss/clock hashes, Git SHA, and explicit target/role strings.

The runtime refuses schema/order mismatches, physical-hull extrapolation, and
enabled invariant corrections when no coefficient is deployed. Feature-domain
departures are counted rather than silently clipped. Every variational run
returns an explicit `runtime_gate` decision with the zero-repair, OOD, and
closure-overhead limits and all failure reasons.

## Runtime modes

The production candidate is selected as one coupled mode:

```yaml
microscopic_closure:
  routing: variational_v2
  angular: variational_v2
  artifact: models/microscopic_closure_v2/closure_v2.npz
  invariant_corrections: true
```

It preserves the NTC clock and scalar loss. Positive post-collision modal
energies follow by construction; a non-positive energy raises an error rather
than triggering a repair.

The runtime still samples the retired gated kernel. Wiring the conditional
kernel in requires the two-dimensional `(a, u)` quantile table, where
`a = lambda3 z_in + lambda4 eps` is the scalar the conditional law depends on;
until that lands, `build_artifact` refuses to export an energy sampler for any
node with a non-zero `lambda3` rather than silently dropping the memory term.

The complete v1 path is still:

```yaml
microscopic_closure:
  routing: legacy_rank0
  angular: legacy
```

The prior schema-2.1 `ctc_moment16/ctc_vss_rank2` path is also retained for
historical A/B runs. The legacy conditional GMM is neither loaded nor called by
`variational_v2`; its inelastic-target audit is recorded in
`reports/cond_gmm_audit.md`.

## Build and verification

```bash
bash hpc/setup_negishi_env.sh
make -C HS_CTC_v2/build clean all
make test
```

### Generator cost

No force acts before first contact, so the approach is integrated by
conservative advancement: the axis-to-axis gap cannot close faster than
`g + (|w1| + |w2|) L/2`, so stepping by `(gap - D)` over that bound provably
cannot skip a contact, and the step is exact rather than approximate because
translation is linear and each director precesses about a fixed axis. The
frozen fixed-step scheme still runs from just before contact through the
contact itself.

This removes 99.996% of the integration steps. At AR = 3 the generator does
about 714 hits per second per core against 0.12 before, so the planned
5.1e8-event campaign moves from roughly 325,000 core-hours to about 200. The
accepted/rejected set is bit-identical; outcomes differ only by the contact
discretisation both schemes share, verified against a reference at 64 times the
contact resolution. Set `CTC_FAST_APPROACH=0` for the old behaviour.

The test suite includes all 28 reference numerical checks under the corrected
projection/rate semantics. It also covers binary compatibility, projected
area/ESS, quantile sampling, artifact validation, positive-energy collision
updates, NTC regression, and the frozen legacy seeded HCS golden output.

## Restarting on Negishi

First cancel only this project's CTC arrays while preserving all output:

```bash
bash hpc/cancel_ctc_jobs.sh --cancel
```

The command creates a timestamped directory under `reports/operations` with
the pre/post queue, accounting history, Git SHA, manifest snapshot, `_SUCCESS`
inventory and count, all run metadata, and cancelled array-parent IDs. It
leaves every completed run in place and moves each incomplete directory intact
under `results/quarantine/ctc_cancel_<timestamp>` so a fresh run cannot
overwrite it. Nothing is deleted.

Then build, generate the manifest on the login node, and use the conventional
Negishi submission for the 36-node, 5,000-hit sentinel:

```bash
make -C HS_CTC_v2/build clean all
hpc/python.sh hpc/make_closure_manifest.py \
  --stage sentinel --samples 5000 --output manifests/closure_sentinel.csv
sbatch job_closure_sentinel.slurm
```

Each completed shard includes `runtime_v2.json`, so budgets use measured
Negishi hits/second and attempts/hit. When all 36 `_SUCCESS` markers exist:

Completed target directories are skipped. An existing target without
`_SUCCESS` causes the task to fail closed instead of overwriting partial data;
move that directory to `results/quarantine` before resubmitting it.

```bash
FIT_JOB=$(sbatch --parsable job_estimate_closure_sentinel.slurm)
FIT_JOB=${FIT_JOB%%;*}
sbatch --dependency=afterany:"$FIT_JOB" job_summarize_closure_sentinel.slurm
cat results/closure_estimates/sentinel_report.json
```

Use `afterany`, not `afterok`, for the sentinel summary. An infeasible moment or
projection is a scientific gate result that must appear in the report, rather
than leaving the summary permanently pending with `DependencyNeverSatisfied`.
Missing or corrupt binary input still makes the fit task fail hard, and the
summary refuses release when any of the 36 node estimates is absent.

The sentinel axes are `alpha={0.5,0.8,0.95,1}`, `theta={0.2,1,2}`, and
`AR={1.1,2,3}`. Do not submit the full grids unless
`all_sentinel_gates_pass` is true.

The full baseline manifest is generated with:

```bash
hpc/python.sh hpc/make_closure_manifest.py \
  --stage baseline --samples 5000 --output manifests/closure_baseline.csv
```

It contains the requested 13 alpha values, 16 theta values through 2.0, and
six aspect ratios. Compatible completed schema-2.1 baseline shards may be
included by the read-only loader instead of being regenerated.

The excitation design generator enumerates 47 direct ensembles on the coarse
`3 x 4 x 3` grid plus `(1,1,2)` elastic sentinel:

```bash
hpc/python.sh hpc/make_closure_manifest.py \
  --stage excitation --samples 5000 --output manifests/closure_excitation.csv
```

Execution of nonzero `ensemble_id` is intentionally release-gated in the CTC
binary until the baseline sentinel has passed and the direct sampler is
certified. This prevents thousands of mislabeled baseline runs from being
accepted as excitation data. Adaptive continuation is capped at 200,000 hits
per ensemble and is driven by the 95% half-width of `beta_n X_n`.

## Release gates

Artifact or production release requires:

- predicted acceptance matching the observed hit fraction within three standard
  errors or 2 percent, whichever is looser, since the z-test necessarily
  tightens as the event count grows;
- inverse-propensity `ESS/N >= 0.5` and proposal balance within three standard
  errors on all 14 invariants;
- raw `0 < p_exch <= 1` for the affine memory diagnostic and both projection
  residuals below `1e-6`;
- less than `0.02` nats of held-out log-density gained by adding the quadratic
  memory statistic `z^2 z'`;
- equilibrium `Beta(2,2)` invariant law at `alpha=1`, at every theta and
  aspect ratio, within the looser of three bootstrap standard errors and 2%,
  so a weakly identified node cannot buy a pass with a wide error bar and a
  well resolved one is not failed for a negligible offset;
- sampled `<z_in>` matching its exact law `p(z) ~ z(1-z)(z/theta + 1 - z)^-4`,
  which is *not* `theta/(theta+1)`: that is the ratio of the means and is low
  by 22% at `theta=0.2`;
- numerical sampler moment error below `1e-3`;
- one fixed point in the calibrated theta domain with negative numerical
  drift derivative, including fitted-surface derivatives;
- zero negative-energy repairs, feature OOD frequency below 0.1%, and closure
  overhead below 5%;
- held-out HCS/USF validation without refitting, targeting 2% HCS `theta*` and
  a demonstrable reduction of the legacy USF stress error.

Failures are reported and block the next campaign stage; they are never
converted into clipping, silent extrapolation, or parameter replacement.
