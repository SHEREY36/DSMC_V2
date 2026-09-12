# DSMC coupling resolution and remaining work

**Branch:** `closure-v2-repairs`  
**Investigated from:** `05ed1d6`  
**Date:** 2026-09-07

This report supersedes the open coupling diagnosis in
`handoff_dsmc_coupling.md`.  The Negishi baseline/refit campaigns were not
modified and may continue running.

## Result

The CTC closure and the DSMC collision-selection measure were not the cause of
the low HCS temperature ratio.  The runtime independently interpolated the
natural parameters, each node's `a` coordinate, and the conditional quantile
table, then evaluated the resulting synthetic table.  These nonlinear
operations do not commute.  At `alpha=0.95, AR=2` that construction created
three drift roots,

```text
0.907836, 1.017500, 1.233763,
```

although the fitted CTC operator has one stable root near 0.99.  Runtime now
evaluates each neighbouring fitted node first and then interpolates its
conditional quantiles with one common physical interpolation stencil.

On the existing 24-node artifact the corrected deployed operator has:

| alpha | AR | predicted theta* | drift derivative |
|---:|---:|---:|---:|
| 0.95 | 2 | 0.984571 | -0.13827 |
| 0.95 | 3 | 1.003149 | -0.25711 |
| 1.00 | 2 | 1.000286 | -0.24467 |
| 1.00 | 3 | 1.001555 | -0.25851 |

All are unique stable roots.  A local 2,003-particle HCS run at
`alpha=0.95, AR=2`, with invariant corrections disabled, ended after 10.001
collisions per particle at `theta=0.986648`.  Its collision audit measured a
mean state `theta=0.985041`, realized drift `-1.015e-3`, and conditional-law
predicted drift `-5.988e-4`.  There were no negative-energy repairs or
out-of-domain queries, and the runtime gate passed.

## Why the interpolation order changes the physics

At fitted node `i`, the energy kernel is represented by the conditional
quantile

```text
Q_i(u | z, eps) = Q_i(u; a_i),
a_i = lambda1_i + lambda3_i z + lambda4_i eps_i.
```

For physical interpolation weights `w_i`, the old runtime used, schematically,

```text
Q_old = Interp(Q_i tables) evaluated at
        Interp(lambda_i) and Interp(a-grid_i).
```

There is no identity that permits this because `Q` is nonlinear in both its
natural parameters and its tabulated coordinate.  The repaired runtime uses

```text
Q_runtime(u | z, eps) = sum_i w_i Q_i(u | z, eps_i).
```

This is node-first quantile interpolation.  It stays monotone and is the
one-dimensional Wasserstein interpolation of the neighbouring conditional
laws.  All physical quantities in a kernel state use the same interpolation
vertices and weights.  A complete Cartesian cell uses multilinear weights, so
a query exactly on a sampled alpha or AR plane cannot borrow a different
plane because of an arbitrary Delaunay diagonal.  Incomplete cells fall back
to a single fail-closed barycentric stencil.

The artifact stability gate now integrates the exact exported quantile law,
not an analytic surrogate constructed by separately interpolating parameters.

## Checks that separated the causes

For a selected pair with incoming translational fraction `z`, total fractional
loss `eps`, and outgoing fraction `z'`, the instantaneous HCS ratio drift is
proportional to

```text
D = (2/3)[(1-eps) z' - z]
    - theta[(1-eps)(1-z') - (1-z)].
```

Consequently the correct diagnostic is energy-weighted collision drift, not
an unweighted mean partition and not `theta=z/(1-z)`.

The following were checked independently:

1. The production DSMC selection gave accepted `<z>=0.48615` and
   energy-weighted `<z>=0.48530` for the elastic AR=2 ruler, against the raw CTC
   value 0.48597.  Pair selection is therefore correct at the precision needed
   here.
2. Evaluating the fitted analytic bridge on the empirical joint CTC
   `(z, eps, E)` distribution reproduced the raw CTC drift.  The fitted closure
   itself is not losing the target response.
3. Scalar loss has two separate jobs.  The frozen BL draw determines the
   surviving energy `(1-eps)E`; the bridge's `lambda4 eps` covariate must be put
   on the CTC loss scale on which `lambda4` was fitted.  The gate and runtime
   now preserve this separation.
4. The reported elastic `+4.8%` discrepancy was a late-window average over one
   correlated fluctuation.  The same trajectory ended at 1.004.  It was not a
   systematic elastic fixed point.

## Initial-state repair

The legacy scalar path set `omega_x=0` and subsequently chose a rod axis
perpendicular to that vector.  Once tensor invariants became active this was
not isotropic: it generates an order-one preferred laboratory plane and an
initial `RtRt` around 0.1.  Variational runs now draw an isotropic axis and
project a three-dimensional Gaussian angular velocity into its tangent plane.
This is exactly a two-degree-of-freedom Maxwellian conditional on the rod axis.
The legacy initialization and seeded regression path are unchanged.

## Reweighted excitation: mathematical status

Fresh CTC trajectories are not mathematically required merely to change the
incoming one-particle distribution.  Under molecular chaos, let

```text
f_eta(x) = r_eta(x) f_0(x) / Z_eta.
```

The accepted baseline collision density contains

```text
f_0(x1) f_0(x2) k_hit(x1,x2,b) P(Gamma | x1,x2,b).
```

The target-to-baseline Radon--Nikodym derivative is therefore proportional to

```text
r_eta(x1) r_eta(x2).
```

Both the geometric hit factor and the conditional contact trajectory cancel.
Thus any collision observable can be estimated from existing accepted events:

```text
E_eta[F] = E_0[r_eta(x1) r_eta(x2) F]
           / E_0[r_eta(x1) r_eta(x2)].
```

This is exact importance sampling for a target distribution absolutely
continuous with respect to the baseline.  It is not synthetic outcome
generation.  It fails only when overlap is poor, which is measured by the
effective sample size and maximum normalized weight.  Fresh CTC is required
for failed-overlap directions, a changed contact law, missing support, or as an
independent validation sample.

The code now joins each pair ratio to an accepted outcome by the recorded
`(event_id, attempt_index)` key and passes it into node estimation.  Virtual
excited nodes carry the achieved cell-measure invariants and are accepted by
the artifact builder only when they point back to the corresponding baseline
shard.

Two errors in the earlier excitation implementation were corrected:

- `A_cu` is `(c.u)^2-c^2/3`, not a radial `c^3` moment.
- `W2` is the U-statistic `|<omega>|^2`, not axial spin
  `<(omega.u)^2>`.  Smooth rods enforce `omega.u=0`, but can still have a
  nonzero mean tangent spin in a laboratory direction; `W2` is identifiable.

Signed paired directions were added for `PiQ`, `PiRt`, `QRt`, and
`qtr_qrot`.  Changing the sign of a single joint tilt flips both component
means and cannot reverse their contraction, so parallel and opposed score
directions are both necessary.  On 100,000 real AR=2 attempts, examples at
`|eta|=0.4` retained ESS about 0.73 and reached `A_cu=+0.207/-0.180`,
`W2=0.050`, `PiQ=+0.013/-0.013`, and `qtr_qrot=+0.076/-0.068`.

## Near-sphere interpretation

AR close to one is a singular physical limit, not merely a regression
degeneracy.  At AR=1.1 the measured exchange probability is only about 0.05
and `lambda3` is 170--350, so the conditional kernel is close to the identity
and its HCS attractor can lie below the old `theta=0.2` grid.  The low-theta CTC
campaign is therefore necessary before invoking cumulant corrections.

For homogeneous isotropic cooling, tensor and heat-flux invariants vanish in
the infinite-particle ensemble.  They cannot systematically repair the AR=1.1
HCS fixed point.  The possible first-order HCS corrections are the isotropic
scalars `a2_tr`, `a2_rot`, `a11`, and `A_cu`.  They should be tested only after
the baseline low-theta kernel brackets a unique root.  For USF/Fourier flows,
the tensor and vector invariants become essential.

## Remaining sequence

1. When the Negishi outputs arrive, verify completeness, one equilibrium
   anchor per AR, `incoming_law`, `incoming_law_energy`, `cell_features`, and
   all QA/precision flags before replacing any artifact.
2. Rebuild the artifact with the node-first stability gate.  Do not deploy if
   any `(alpha,AR)` has zero or multiple roots or if its uncertainty interval
   reaches the theta hull.
3. Repeat the HCS matrix with corrections disabled: elastic ruler first, then
   AR=2/3 DEM points, then AR=1.5/2.5 interpolation checks, and finally the
   low-theta AR=1.1/1.2 cases.
4. Estimate the four isotropic HCS excitation responses by importance
   sampling.  Use fresh CTC only for a direction whose attempt- and outcome-ESS
   gates fail, then compare one fresh excited node as an external validation.
5. Before enabling all-flow first-order corrections, regress and test all
   affected natural parameters.  Restricting every response to `lambda1` is a
   modeling assumption, not a consequence of the variational formulation;
   near the sphere, changes in exchange strength and memory (`p_exch` and
   `lambda3`) must be tested explicitly.
6. Only after HCS passes end to end, proceed to USF and then Fourier/thermal
   gradient validation.  Those flows exercise anisotropic stress, orientation,
   and heat-flux features that HCS cannot identify dynamically.

