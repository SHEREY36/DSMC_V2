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
- the v1 CTC normal damping based on centre translational relative velocity.

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

All proposals estimate the DSMC incoming-state measure. Accepted outcomes use
weights proportional to `1/A_perp`; the common normalization is immaterial.
The observed hit flag is calibrated against `A_perp/A0`. The frozen DSMC
cross-section polynomial is reported separately as a clock audit and is never
required to equal a geometric area.

The energy kernel is

```text
K(z | z_in) = (1 - p_exch) delta(z - z_in) + p_exch R(z),
R(z) proportional to Beta(2,2)(z) exp(lambda1 z + lambda2 z^2).
```

`p_exch` is identified directly from the weighted affine memory regression;
there is no `2/Z` multiplier and no probability clipping. `lambda1` and
`lambda2` solve the exact convex I-projection for the reset mean and ordinary
second moment. State corrections act in the natural parameter,
`lambda1 = lambda1_0 + beta dot X`.

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

It preserves the NTC clock and scalar loss. If the exchange gate stays closed,
both the translational partition and individual rotational split are
preserved. If it opens, the tilted reset law and a uniform `Beta(1,1)`
rotational sub-split are drawn. Positive post-collision modal energies follow
by construction; a non-positive energy raises an error rather than triggering
a repair.

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
sbatch job_estimate_closure_sentinel.slurm
sbatch job_summarize_closure_sentinel.slurm
cat results/closure_estimates/sentinel_report.json
```

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

- propensity agreement within three standard errors and inverse-area
  `ESS/N >= 0.5`;
- raw `0 < p_exch <= 1`, feasible reset moments, and both projection residuals
  below `1e-6`;
- less than 2% held-out improvement from a nonlinear memory regression;
- equilibrium `Beta(2,2)` reset at `alpha=1, theta=1`;
- numerical sampler moment error below `1e-3`;
- one fixed point in the calibrated theta domain with negative numerical
  drift derivative, including fitted-surface derivatives;
- zero negative-energy repairs, feature OOD frequency below 0.1%, and closure
  overhead below 5%;
- held-out HCS/USF validation without refitting, targeting 2% HCS `theta*` and
  a demonstrable reduction of the legacy USF stress error.

Failures are reported and block the next campaign stage; they are never
converted into clipping, silent extrapolation, or parameter replacement.
