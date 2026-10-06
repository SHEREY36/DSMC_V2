# Energy kernel: from the gated Beta to a conditional I-projection

Date: 2026-09-02. Scope: `Coll_Models_v2` estimation only. The DSMC runtime
still samples the retired kernel; see "What is not done" below.

## Why

The gated form

    K(z' | z) = (1 - p_exch) delta(z' - z) + p_exch R(z')

makes a checkable prediction: a point mass at `z' = z` carrying `1 - p_exch`.
The sentinel events do not have one.

| AR | fitted `p_exch` | mass the model needs at `dz ~ 0` | observed `|dz| < 0.01` | observed `sd(dz)` |
|----|-----------------|----------------------------------|------------------------|-------------------|
| 1.1 | 0.054 | 94.6 % | 13.2 % | 0.075 |
| 2.0 | 0.712 | 28.8 % |  3.7 % | 0.269 |
| 3.0 | 0.747 | 25.3 % |  3.7 % | 0.280 |

(elastic replay, alpha = 1, theta = 1)

Two failure modes follow.

1. **Variance floor.** The gate forces a conditional variance of at least
   `p(1-p)(z - mu)^2`. At (alpha=1, theta=0.2, AR=1.1) that floor is 0.0057
   while the observed conditional variance is 0.0039, so solving for the reset
   variance returns a negative number.
2. **Divide by p.** `reset_mean = intercept / p_exch`. At (alpha=0.5,
   theta=2.0, AR=1.1) the fitted `p_exch` is 0.011, so a -0.10 intercept
   becomes a reset mean of -8.99. The intercept is an extrapolation of the mean
   map to `z = 0`, far outside the data.

Eight of the 36 sentinel nodes raised `ValueError` for one of these reasons,
all at AR = 1.1. The model form is wrong at every node; AR = 1.1 is only where
the arithmetic gives up.

## What changed

The reference kernel is now the *memoryless* Borgnakke-Larsen draw, and the
sufficient statistics gain the memory and fractional-loss terms:

    p(z' | z, eps) ~ Beta(2,2)(z')
                     * exp(lambda1 z' + lambda2 z'^2 + lambda3 z z' + lambda4 eps z')

Same I-projection theorem, no atom, no variance floor, and a strictly convex
dual on sample moments, so the infeasible-moment branch cannot occur. `lambda4`
lets the runtime evaluate the kernel at its own frozen BL loss instead of
inheriting the CTC's; it is deployed only when the loss actually varies, so it
is identically zero at alpha = 1.

Files: `projections.py` (new `fit_conditional_energy_projection`,
`conditional_energy_logpdf`, `conditional_energy_mean_map`,
`conditional_energy_stationary`, `incoming_partition_density`),
`fit_exchange.py` (rewritten), `estimate.py` (carries the per-event loss,
warm-starts the bootstrap), `artifact.py` (theta drift), `pipeline.py` (gates).

## Semantics that moved

* `reset_mean` / `reset_second_moment` keep their names but now report the
  first two moments of the kernel's **invariant** law. That is the object the
  reset law stood for, and it makes the elastic gate a direct test that an
  elastic kernel reaches equipartition (Beta(2,2): mean 1/2, second moment
  3/10) at every theta and aspect ratio, not only at theta = 1.
* `p_exch` is now a reported diagnostic, never a parameter and never a veto.
  It is still unclipped; a value outside (0,1] sets `memory_diagnostic_pass`
  false rather than aborting a projection that is well posed.
* `model_form_pass` compares held-out log-density against a family enriched
  with `z^2 z'`, with a 0.02 nat tolerance, instead of comparing mean-map MSE
  against a Legendre regression.
* The theta drift in `_stability_rows` now uses the fitted kernel's `E[z_out]`,
  averaged over the exact collision-weighted incoming law
  `z(1-z)(z/theta + 1 - z)^-4`, instead of the Bernoulli composition.

## Known limits, measured

* On synthetic gated-Beta data the invariant law is recovered to 0.001 at
  `p_exch = 0.4`, 0.012 at 0.65 and 0.028 at 0.2. The gate is not a member of
  this family and the weaker the exchange the fewer collisions carry
  information about the law being approached. This bounds how much to trust the
  elastic gate at AR = 1.1, where the memory slope is about 0.94.
* `lambda3` alone is not a clean memory dial: with `lambda1` and `lambda2` at
  zero the tilt also drags the marginal towards `z = 1`, and the mean-map slope
  is non-monotone in `lambda3`. What the estimator guarantees is that the
  fitted kernel reproduces the observed lag-one slope exactly, because
  `E[z z']` is a matched moment. Separating rate from target cleanly needs the
  Sinkhorn-normalised bridge form, where row normalisation and Beta(2,2)
  stationarity are the same equation and equipartition holds for every
  `lambda3` by construction.
* Cost: about 1 s per 5,000-event node fit and 60 s at 300,000 events, with the
  working set bounded by `CONDITIONAL_ENERGY_CHUNK x quadrature`.

## What is not done

* The DSMC runtime still samples the gated kernel. Wiring in the conditional
  form needs the two-dimensional `(a, u)` quantile table, where
  `a = lambda3 z_in + lambda4 eps` is the only event-dependent quantity the
  conditional law sees. Until then `build_artifact` raises rather than export
  an energy sampler that silently drops the memory term.
* The measure repair is now done; see the second half of this file.

---

# Measure: from the static shadow to the kinematic acceptance

Date: 2026-09-02. Scope: `Coll_Models_v2` weighting. Same shards, no new CTC data.

## Why

`weights.py` weighted accepted outcomes by `1/A_perp`, the projected excluded
area of a *frozen* pair. That is not the generator's acceptance probability.
The rods turn while they close, so the acceptance is the dynamic excluded area.

| AR | `<A_perp>` | observed area, theta = 0.2 | theta = 1 | theta = 2 |
|----|-----------|----------------------------|-----------|-----------|
| 1.1 | 3.459 | 3.508 | 3.479 | 3.453 |
| 2.0 | 6.677 | 7.993 | 7.284 | 7.050 |
| 3.0 | 11.00 | 14.684 | 12.759 | 11.940 |

Binning proposals by the rotation number `Omega = |omega| (L+D) / g` gives a
clean monotone dose-response, and the lowest-Omega bin sits at 1.007: the
formula is exactly right, it is just the wrong object.

This is physics, not a staging artefact. Contact requires a centre separation
below `L + D`; the generator starts beyond that; and free rotation maps an
isotropic director law to an isotropic one. So the answer cannot depend on the
staging distance, and a force-free kinematic model confirms it does not --
quadrupling the run-up moves the effective area by less than the Monte-Carlo
error.

## What changed

`encounter_propensity` integrates the force-free encounter over the impact
plane: the pair closes at constant speed along `ghat` from the staging
distance while both rods turn freely, accepted if the axes ever come within
`D`. Offsets use a randomised sunflower lattice on the disc of radius `L + D`,
which is the only region where contact is possible. Blocks are ordered by
rotation number and closed on a work budget, because that number is heavy
tailed -- median near 8 radians, tail past 100 -- and a few fast rotators would
otherwise set the resolution for every event.

Accepted outcomes are then weighted by `debiased_inverse(propensity)`. The
plug-in `1/p_hat` carries a Monte-Carlo inflation of `(1-p)/(M p)`, running
from 0.1 percent at `p = 0.8` to 5 percent at `p = 0.1`, which does not cancel
in normalised weights; the leading term is removed in closed form.

## Validation

* **Zero-spin limit.** With `omega = 0` the integrator reproduces the analytic
  `A_perp` to 0.03 percent in the mean. This pins the geometry, the disc
  normalisation, the staging and the segment-distance routine at once.
* **Against the generator.** Predicted acceptance versus observed hit fraction:

  | node | static `1/A_perp` | kinematic |
  |------|-------------------|-----------|
  | AR 3, theta 2 | +7.87 % | −0.52 % |
  | AR 3, theta 0.2 | +25.06 % | +0.30 % |
  | AR 2, theta 2 | +5.24 % | −0.20 % |

* **Proposal balance**, largest \|z\| over the 14 invariants: 3.74 -> 0.70,
  3.09 -> 1.20, 3.62 -> 1.33. Effective sample size rises slightly.
* **Step convergence** at the worst node (AR 3, theta 0.2, median turn 7.8
  radians): 0.3994, 0.4004, 0.4007, 0.4008, 0.4008 at 48, 96, 192, 384 and 768
  steps. The deployed rule of 24 steps per radian sits above the knee.

## Gate semantics

The hit-propensity test stays a genuine test rather than a tautology, because
the propensity is an independent physical calculation and not a model fitted to
the hit flags. Alongside the z-score, which necessarily tightens as the event
count grows, the relative bias is now reported so the tolerance can be stated
as a physical accuracy: the gate passes on three standard errors *or* 2 percent,
whichever is looser.

## Cost, and what to do about it

45 to 110 seconds per sentinel node at 128 offsets over all attempts. Two
things would make this cheap at campaign scale:

* the proposals are shared across the four alpha values at a given
  `(theta, AR)` by common random numbers, so the same propensity is currently
  computed four times;
* the same ray test belongs in the generator, where it costs a fraction of one
  DEM flight and removes the post-processing step entirely.

---

# Gates: three that were testing the wrong thing

Date: 2026-09-02. Scope: `Coll_Models_v2` QA. No refitting; the numbers do not
move, only the pass/fail verdicts.

## Elastic limit: from a flat tolerance to an error-aware one

The gate compared the kernel's invariant law against Beta(2,2) with a flat
`0.02`. Two sentinel nodes sat either side of that line for reasons the
tolerance could not distinguish:

| node | invariant second moment | bootstrap error | sigma | flat 0.02 | error aware |
|------|------------------------|-----------------|-------|-----------|-------------|
| alpha=1, theta=0.2, AR=2.0 | 0.3222 | 0.0111 | 2.0 | fail | pass |
| alpha=1, theta=0.2, AR=1.1 | 0.5811 | 0.0741 | 3.8 | fail | fail |

The AR=1.1 node has `Z_rot = 16.5`, so its kernel is nearly the identity and
its invariant law is inferred from a small residue. That is a genuine
identifiability failure and it should be reported. The AR=2.0 node is well
resolved and two standard errors out. The gate now allows the looser of three
bootstrap standard errors and 2 percent, so a badly resolved node cannot buy a
pass with a wide error bar, and a well resolved node is not failed for a
physically negligible offset.

## Incoming partition: the plan's formula is the wrong average

Gate G-C in the implementation plan asks for `<z_in> = theta/(theta+1)` to
1 percent. That is the ratio of the means; `z` is the mean of a ratio. Both
modal energies are Gamma(2) under the generator, so

    p(z) proportional to z (1 - z) (z/theta + 1 - z)^-4.

At theta = 0.2 the exact mean is 0.2132 while `theta/(theta+1)` gives 0.1667, so
the plan's gate would report a 22 percent failure on a correct sampler. The
gate is now implemented against the exact law, and the generator passes it:

| theta | exact mean | sampled | relative difference | z |
|-------|-----------|---------|---------------------|---|
| 0.2 | 0.21322 | 0.21218 | -0.49 % | -0.69 |
| 1.0 | 0.50000 | 0.49930 | -0.14 % | -0.38 |
| 2.0 | 0.63553 | 0.63515 | -0.06 % | -0.22 |

The spreads match to three digits as well. The same diagnostic also reports the
inverse-propensity-weighted mean over accepted outcomes, which tests the measure
conversion rather than the generator.

## Propensity: a significance test cannot express a physical tolerance

A z-score on the mean residual tightens without bound as the event count grows,
so at campaign scale it would reject a geometrically perfect model. The gate now
passes on three standard errors *or* 2 percent, whichever is looser, and reports
the relative bias alongside the z-score.

---

# Generator cost: conservative advancement

Date: 2026-09-02. Scope: `HS_CTC_v2` integrator. No physics changed.

## Why

`dt = TCOLL/50` was applied to the whole trajectory, including the approach from
the staging distance. At AR = 3, theta = 1 that is about 1.5 million steps of
force-free straight-line motion against roughly 50 steps of actual contact:
**99.996 percent of the integration steps compute nothing.** The signature is in
the runtime files, where attempts per second scale as `sqrt(theta)`, which is
the flight time and nothing else.

## What changed

No force acts before first contact, so translation is exactly linear and each
director precesses at a constant rate about a fixed laboratory axis: a large
step is *exact*, not approximate. The axis-to-axis gap cannot close faster than
the relative centre speed plus the fastest surface speed rotation can
contribute, so

    dt_free = (gap - D) / (g + (|w1| + |w2|) L/2)

provably cannot skip a contact. The step is taken only while it exceeds the
fixed step, so the approach to contact and the contact itself keep the frozen
scheme untouched. The large step uses an exact Rodrigues rotation about the
angular-velocity axis, because the fixed-step path's split quaternion is only
first order in the step.

`CTC_FAST_APPROACH=0` restores the old behaviour; `CTC_DT_DIVISOR` exposes the
steps per contact time, whose frozen v1 value is 50.

## That it changes no physics

The accepted/rejected set is bit-identical across the two schemes at both theta
values tested. Post-collision quantities differ by about 1.6 percent per event,
which is *not* the approach scheme: refining the contact resolution against a
converged reference shows both paths are the same distance from it.

| scheme | steps per contact | median error in `delta_total` |
|--------|-------------------|-------------------------------|
| fast approach | 50 | 1.62e-2 |
| fast approach | 200 | 4.40e-3 |
| fast approach | 800 | 1.12e-3 |
| frozen v1 fixed step | 50 | 1.71e-2 |

(reference: fast approach at 3200 steps per contact; 250 events, AR = 3,
theta = 1, alpha = 0.8)

The errors fall by four for each fourfold refinement, which is first order, and
the frozen v1 setting sits exactly where the fast approach sits at the same
resolution. Ensemble means agree to 0.15 percent in `<delta_total>` and 2e-5 in
`<cos chi>`.

**Worth noting on its own:** the frozen 50 steps per contact carry a 1.7 percent
per-event and 0.5 percent ensemble-mean discretisation error in the dissipated
energy. That was previously unaffordable to improve; it now costs nothing.

## Cost

| configuration | 250 hits, AR = 3, theta = 1 |
|---------------|------------------------------|
| frozen v1 fixed step, 50 steps/contact | 327 s |
| fast approach, 50 steps/contact | < 1 s |
| fast approach, 3200 steps/contact | 1 s |

At production scale, AR = 3 with 20,000 hits on four threads: 7 s at theta = 0.2,
8 s at theta = 1 and 2. The cost is now essentially theta independent, which
confirms the flight was the entire theta dependence.

Per core that is about 714 hits/s against 0.12 hits/s on Negishi, so the planned
campaign moves from roughly 325,000 core-hours to about 200 -- below the 1,400
the implementation plan originally assumed.
