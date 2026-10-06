# Spherocylinder HCS non-Gaussian study - results and readings
### 2026-09-27, protocol hcs-ng-v8, analysis hcs-ng-analysis-v3

Campaign: 360 tasks, 36 coordinates, N = 10^4, tau in [500,1500] with
delta-tau = 5, ten realizations per coordinate split five-and-five across two
bracketing initial temperature ratios. Artifact
`f6a6cbec...e3f616`. Every stage from the numerics pilot upward passed.

---

## 1. Data audit - independent of the campaign's own gates

Re-derived from the raw trajectories rather than by re-reading a verdict:

| check | result |
|---|---|
| non-finite / negative temperature / theta runaway | **0 of 360 trajectories** |
| two-branch gap in ln theta, late window | max **0.0024**, median 0.0007 (tolerance 0.03) |
| drift across the second half of the window | max **0.0024**, median 0.0011 |
| residual transient inside the window | branch offsets uncorrelated (r = -0.42 to +0.08), \|head-tail\| <= 0.0017 |
| closure fallback in the evaluation window | max **0.0022** (gate 0.01) |
| bulk/thermal temperature ratio | <= **6e-32** (gate 1e-12) |
| NTC majorant violations per accepted pair | <= **1.3e-7** (gate 1e-5) |
| final vrmax / initial | <= **1.025** (gate 3.0) |
| achieved cpp | exactly 1500 on all 360 |

Two points worth keeping:

**The window is on the attractor, and that is a measurement, not an
assumption.** A residual transient would force the two branches to enter the
window from opposite sides, so their head-minus-tail offsets would be
anti-correlated. They are not: r is consistent with zero for all six
observables, and 15-21 of 36 coordinates are anti-correlated, which is what
noise gives. The transient is dead by tau ~ 20-60 (Fig. 1); the first retained
snapshot is at tau = 500.

**The v7 near-sphere closure exclusion at AR = 1.20 was a statistics artifact,
as predicted, and it is gone.** `closure_domain_exclusions` is empty. At
N = 2000 the instantaneous a20 at alpha = 0.5, AR = 1.2 had SE ~ 0.028 against
a hull edge at +0.117, so a +2.2-sigma excursion fell out of domain a few per
cent of the time; at N = 10^4 the SE is 0.013, the same excursion is +4.9 sigma,
and the fallback fraction dropped to 5e-4. AR = 1.10 at alpha = 0.5 remains
genuinely outside the artifact and stays excluded.

Statistical quality: replicate standard errors are **1.3e-4 to 5.8e-4** on
cumulants of order 10^-2, i.e. signal-to-noise of 20-100.

---

## 2. The central claim

> **Elongation and surface roughness are not the same knob.** Aspect ratio
> opens the rotational channel exactly as roughness does - it pulls theta^H
> down from its smooth-sphere divergence - but it simultaneously *reduces* the
> cooling rate and *reduces* every non-Gaussian signature. Geometric coupling
> is milder and more Gaussianizing than frictional coupling.

Megias & Santos (2023) find, for constant-beta rough disks and spheres, that
opening the rotational channel produces a02 of order one, a breakdown of the
Sonine approximation, and strongly algebraic rotational tails. Here the same
channel opening produces cumulants that are **twenty times smaller** and tails
that are, over most of the domain, statistically indistinguishable from a
Maxwellian.

The proposed mechanism (this is inference, not measurement): constant beta
applies the *same* tangential impulse at every impact geometry, so it is a
singular, impact-angle-independent sink. A spherocylinder's tangential coupling
is set by the lever arm at contact, which is distributed from zero (axial or
centre-on impact) to L/2 (tip-grazing). That distribution both averages the
per-collision sink downward and acts as a randomizer over collision types, which
pushes the velocity distribution back toward Gaussian. Two independent
observables carry this signature - the cooling rate and the tail exponents -
which is why it is worth stating as a mechanism rather than two coincidences.

---

## 3. Effective restitution: shape shields dissipation

Following Hong et al. (2022), define alpha_eff as the restitution a *smooth*
sphere gas needs to cool at the observed rate. The cooling rate is recovered
from the energy-weighted per-collision loss L via

  zeta* = -d ln T / d tau = 2 L (1 + theta) / (3 + 2 theta),

which follows from N/2 pair events per unit tau, each removing L of the pair's
internal energy 2(Ttr + Trot), against a total 5T = 3Ttr + 2Trot. **Validated
against Haff's law fitted directly to the freely cooling runs: 0.02 %, 0.11 %,
1.1 % and 2.3 % at the four inelastic engineering coordinates**, the residual
tracking a20 - itself the leading non-Gaussian correction to the cooling rate.
The same accounting for a smooth sphere returns zeta* = (1-alpha^2)/3, exactly
Hong's Eq. (3), so

  **alpha_eff = sqrt(1 - 3 zeta*)**

in the same convention, with no fitted constant anywhere.

| alpha_eff - alpha | AR 1.1 | 1.2 | 1.35 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|
| alpha = 0.50 | - | +0.209 | +0.236 | +0.256 | +0.286 | +0.302 | +0.306 |
| alpha = 0.80 | +0.072 | +0.081 | +0.090 | +0.097 | +0.114 | +0.112 | +0.112 |
| alpha = 0.95 | +0.018 | +0.020 | +0.022 | +0.024 | +0.028 | +0.027 | +0.027 |

**Readings.**

1. alpha_eff > alpha everywhere: a rod at alpha = 0.5, AR = 3 cools like a
   smooth sphere at alpha = 0.81. This is a *large* effect - a third of the way
   back to elasticity - and it is the single number most likely to matter to
   anyone modelling elongated grains with a sphere code.

2. **The shielding saturates near AR ~ 2** and is flat to 3.0. Shape stops
   buying you anything past roughly two diameters. Combined with the
   lever-arm picture this says the collision statistics are already fully
   anisotropy-dominated at AR = 2; adding length changes the cross-section but
   not the distribution of couplings.

3. The excess scales roughly as (1-alpha^2): at AR = 3 the excess is 0.306,
   0.112 and 0.027 for alpha = 0.5, 0.8, 0.95, against (1-alpha^2) ratios
   0.75 : 0.36 : 0.0975 - i.e. it is very nearly a *constant fractional*
   reduction of the dissipation, about 30-40 %, rather than an additive offset.
   That is the form to quote for a closure.

4. The AR -> 1 limit is singular and our grid cannot resolve it. theta^H
   diverges as (AR-1)^p with p = -1.36 to -1.44 fitted on the three smallest
   AR (Fig. 2b), while zeta* is still *rising* toward the smooth value at
   AR = 1.1. Both cannot continue: at AR = 1 exactly, torque vanishes,
   rotational energy is frozen, T is rotation-dominated and zeta* -> 0. So
   zeta*(AR) must turn over somewhere below AR = 1.1. **That turnover is a
   concrete prediction, and it is exactly where the closure artifact runs out
   of domain** - the model's validity boundary and the physical singularity
   coincide, which is not a coincidence and is worth saying explicitly.

---

## 4. Equipartition breaking

theta^H = Trot/Ttr falls monotonically toward 1 with AR and diverges as AR -> 1
(Fig. 2). At alpha = 0.5: 7.35 (AR 1.2) -> 1.53 (AR 3.0). At alpha = 0.8: 4.93
(AR 1.1) -> 1.02 (AR 3.0).

**The reading that matters: the long rod is closer to equipartition than the
near-sphere**, which inverts the naive intuition that the more anisotropic body
should be the more anisotropic gas. AR is the *coupling strength*, not the
anisotropy: it sets the lever arm, so AR -> 1 is the *smooth*-sphere limit
(beta -> -1 in Megias-Santos language), not the rough one. This is the clean
mapping between the two literatures, and it is a geometric roughness with no
friction coefficient anywhere in the model.

---

## 5. The fourth-order cumulants

All values in `hcs_ng_v8_master_table.csv`; Fig. 4.

**a20 is not simply inherited from smooth spheres.** Against the van Noije-Ernst
Sonine value:

| alpha | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|
| measured - IHS (AR = 2) | -0.0163 | -0.0127 | -0.0066 | -0.0003 | +0.0054 | +0.0049 |

The deviation **changes sign at alpha ~ 0.8**: the rotational channel suppresses
the translational kurtosis at strong dissipation and enhances it at weak
dissipation. At alpha = 0.8 the spherocylinder reproduces the smooth-sphere
result to 3e-4 across every AR - a coincidence of the crossing, not an identity,
and worth saying so before a reader concludes a20 is shape-blind.

**a11 changes sign with shape.** At alpha = 0.8 and 0.95 the
translation-rotation velocity correlation crosses zero between AR = 1.5 and 2.0:
positive for near-spheres (fast particles also spin fast), negative for long
rods (fast particles spin slowly). At alpha = 0.5 it stays positive throughout.
Megias & Santos find a11 > 0 everywhere for rough disks and spheres, so **the
negative-a11 region is a spherocylinder effect with no rough-particle
counterpart.** Suggested mechanism: at large AR the collision transfers
translational energy into rotation efficiently, so a recently collided rod has
high w and depleted c - an anti-correlation that weak coupling cannot build.

**a02 is non-monotone in both parameters**, with a minimum near AR ~ 1.2-1.5 and
a second minimum in alpha near 0.7-0.8. Magnitudes stay within +-0.07,
i.e. **about twenty times smaller than the O(1) values that signal the Sonine
breakdown for rough disks.** The Sonine approximation is in no danger anywhere
in this domain - a substantive difference from the rough-particle literature and
one of the paper's cleaner negative results.

---

## 6. Observables with no rough-sphere analogue

A sphere has no body axis, so neither of these exists in Megias-Santos. Fig. 5.

### 6a. Velocity-axis alignment, A_cu = <(c.u)^2 - c^2/3>

**A_cu > 0 everywhere: rods preferentially translate along their own long
axis.** Physically a rod moving axially presents its smallest cross-section,
collides less often, and keeps its speed longer - a kinetic self-selection
effect, the HCS counterpart of anisotropic mobility.

Two facts make this the most interesting single observable in the study:

1. **It is a function of shape alone.** Over 0.5 <= alpha <= 0.95 it varies by
   about 10 % while AR changes it by a factor of 15. Fitted through the origin,

   **A_cu = kappa (AR - 1),  kappa = 0.00734 +- 0.0003**

   with 5-10 % residuals and kappa varying only 12 % across the whole inelastic
   range. A one-parameter law with the correct AR -> 1 limit built in.

2. **It vanishes identically at alpha = 1** (measured <= 3e-4 at every AR, with
   the exact elastic block). It must: at equilibrium the Maxwell-Boltzmann
   distribution is isotropic in c independent of u. So the alignment is
   *switched on* by dissipation but its *magnitude* is set by geometry.

   That combination has a sharp consequence. At alpha = 0.95 A_cu is already at
   full strength; at alpha = 1 it is zero. The entire rise happens inside
   alpha in (0.95, 1), which our grid does not resolve. **Whether A_cu
   approaches zero linearly in (1-alpha) or with a steeper/singular law is an
   open question and a cheap, well-posed follow-up** - a short campaign at
   alpha in {0.96, 0.98, 0.99, 0.995} at two aspect ratios would settle it.

### 6b. Velocity-spin alignment, A_cw = <(c.w)^2 - c^2w^2/3>

A_cw < 0 everywhere: c and w are preferentially perpendicular. Part of that is
forced - a linear body's w lies in the plane normal to u, and c leans toward u -
so pure geometry predicts A_cw = -<w^2> A_cu / 2 = -A_cu/2 in reduced units.

**Measured A_cw is two to four times larger than that.** Decomposing:

| dynamical share of A_cw | AR 1.1 | 1.35 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| alpha = 0.80 | 0.72 | 0.76 | 0.74 | 0.71 | 0.63 | 0.53 |

So at near-sphere the anti-alignment is **80 % a genuine translation-rotation
dynamical correlation**, and only at large AR does axis geometry take over. The
two observables are therefore not redundant: A_cu measures shape-induced
alignment, A_cw carries an independent correlation that survives after the
geometry is removed. Report the decomposition, not just A_cw.

---

## 7. High-velocity tails - and why the usual fit does not survive its own control

This is the section where having an exact elastic control changes the
conclusion. Fig. 6.

The alpha = 1 arm runs the exact elastic block and is a *Maxwellian by
construction*. Fitting the same tail forms over the same windows to that control
gives:

| channel | Maxwellian null | analytic check |
|---|---|---|
| gamma_c (exponential fit) | **6.554 +- 0.065** | 6.602 |
| gamma_w (power law) | **21.23 +- 0.25** | - |
| gamma_cw (power law) | **5.457 +- 0.023** | - |

The analytic entry is an exp(-gamma c) fit to exp(-c^2) over c in [3,5]: it
returns 6.60, because 2c at the window centre *is* the local log-slope of a
Gaussian. So the null is understood, not merely measured.

**Against that null:**

- Measured gamma_c over the whole inelastic grid is 6.28 +- 0.65, i.e. mostly
  *indistinguishable from a Gaussian*. Only alpha <= 0.6 (plus alpha = 0.8 at
  AR = 1.1) departs by more than 3 sigma.
- The parameter-free kinetic-theory prediction gamma_c = 3 pi / mu20 =
  2.5066/zeta* gives **15 to 170** across the grid - one to two orders of
  magnitude above what is fitted. The fitted exponent is not the asymptotic
  one; it is the Gaussian core.
- **The x = c^2w^2 channel fails its negative control outright**
  (`elastic_tail_negative_control_pass: {c: true, w: true, x: false}`): the
  exact Maxwellian passes the same "asymptotic power law" gate, with
  gamma = 5.44-5.49 and R^2 = 0.993, at all seven aspect ratios. A power-law fit
  to phi_cw over a finite decade cannot distinguish a fat tail from a
  Bessel-function Maxwellian. **Any quoted gamma_cw is therefore
  non-discriminating and must not be presented as evidence of an algebraic
  tail.**

This is the honest version of a caveat Megias & Santos raise qualitatively -
they note the gamma_c and gamma_cw agreement is "mainly qualitative" and that
the fitted exponents may "characterize an intermediate velocity regime previous
to the true asymptotic behaviour". The elastic control turns that into a
quantitative statement with a null and a threshold.

**Net reading.** Tail overpopulation D = 1 - gamma/gamma_Maxwell is significant
only for alpha <= 0.6, is largest in the near-sphere corner (D_c = 0.40 at
alpha = 0.5, AR = 1.2), and **falls with AR** (D_c = 0.084 at AR = 3.0) -
the same Gaussianizing effect of elongation seen in the cooling rate, measured
through a completely independent observable.

---

## 8. What to claim, and what not to

**Claim:**
- alpha_eff and the 30-40 % fractional dissipation shielding, with the
  Haff-law validation (0.02-2.3 %).
- theta^H(AR, alpha), the AR -> 1 divergence, and the AR-as-roughness mapping.
- a20's sign-changing deviation from smooth IHS; a11's shape-driven sign change.
- A_cu = 0.00734 (AR - 1), alpha-independent, zero at alpha = 1.
- The A_cw geometric/dynamical decomposition.
- Tails Gaussian for alpha >~ 0.7; enhancement suppressed by elongation.
- That the Sonine approximation does *not* break down here, unlike rough disks.

**Do not claim:**
- Any asymptotic tail exponent. The c and w fits are intermediate-range; the x
  fit fails its control entirely.
- The AR -> 1 limit itself. State the divergence and the predicted zeta*
  turnover, and say the grid stops at AR = 1.1 because the closure does.
- That A_cu is exactly alpha-independent - it moves 12 %, and the approach to
  zero at alpha = 1 is unresolved.

**Cheapest experiments that would add the most:**
1. alpha in {0.96, 0.98, 0.99, 0.995} at AR = 1.5 and 3.0 - resolves how A_cu
   and the cumulants switch off at the elastic point. ~80 tasks, a few hours.
2. A free-cooling (unscaled) arm at every coordinate - turns the validated
   zeta* relation into a direct measurement and removes the 2.3 % Maxwellian
   weighting assumption at strong dissipation.
3. A smooth-sphere (`sphere-controls`) arm - puts the alpha_eff comparison on
   the same code path rather than on the analytic (1-alpha^2)/3.
