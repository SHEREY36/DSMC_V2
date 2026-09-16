# The Sinkhorn bridge: imposing equipartition instead of inferring it

**Status:** implemented, verified on the 200k sentinel, deployed as the default
kernel form (`sinkhorn_bridge_v2`). The previous form
(`conditional_iprojection_v2`) is retained and selectable.

---

## 1. The problem in one paragraph

The closure needs a kernel that says how a collision redistributes energy
between translational and rotational modes. We write it as a law for the
outgoing partition `z' = E_trans / E_total` given the incoming partition `z`
and the fractional energy loss `eps`. Elastically (`alpha = 1`) there is one
thing we know exactly and independently of any data: repeated collisions must
drive the partition to equipartition, whose law here is `Beta(2,2)` — mean
`1/2`, second moment `3/10`. The old kernel had to *infer* that. It missed:
across the 36 sentinel nodes the fitted invariant mean sat 1 to 3 percent off
equipartition, and at `(alpha=1, theta=0.2, AR=1.1)` it sat at **0.7936**. At
200,000 hits those are 4 to 48 sigma departures. The corresponding
`theta*(alpha=1)` came out 1.012 / 0.946 / 0.918 where the physics demands
exactly 1.

## 2. Why inference was the wrong tool

Under the production kinematic weighting, the weighted mean of the *incoming*
partition recovers `Beta(2,2)` at `theta = 1`:

| node | `<z_in>` static shadow weight | `<z_in>` kinematic propensity |
|---|---|---|
| `alpha=1, theta=1, AR=1.1` | 0.4990 | **0.5000** |
| `alpha=1, theta=1, AR=3.0` | 0.4807 | **0.4996** |

The right-hand column is the production weight, `1 / P(accept | state)` from the
force-free encounter integral at 128 offsets. It confirms `Beta(2,2)` is the
correct reference for this system, as it must be: the generator draws
`E_t ~ Gamma(2, T_tr)` and `E_r ~ Gamma(2, T_rot)` independently.

**Correction.** An earlier draft of this report quoted the left-hand column and
concluded there was a 4 percent incoming-partition bias at `AR = 3`. That was
wrong. Those numbers came from `outcome_weights` called with no propensity,
which returns the deprecated static-shadow weight `1 / A_perp` — retained only
for A/B runs, and documented as missing the rotational enhancement and biasing
the accepted ensemble towards fast-spinning pairs. **There is no
incoming-partition bias.** Any diagnosis built on that 0.4807 — including the
hypothesis that the generator's one-quadrant impact sampling mismatches the
propensity's full-disc integral — is chasing an artifact of the wrong weight.

For the record, that quadrant/disc mismatch is real as a matter of code
(`init_part.f90` draws `b_y, b_z` uniform on `[0, BMAX]` while
`encounter_propensity` integrates the full disc), but it does not bias the
estimator. The rods' azimuth about `ghat` is uniform, so averaging over it
gives `A_++ = A_total / 4` exactly; and `z_in` depends only on
rotation-invariant scalars, so the ratio-estimator's error factor has
conditional mean 1 given everything `z_in` depends on. The measurement above
confirms it.

So the real reason inference was the wrong tool is not a biased sample. It is
that at weak coupling the kernel is close to the identity, so very few
collisions carry information about the law it would reach after infinitely
many — and the invariant law is then an extrapolation far outside the support
of the data. At `(alpha=1, theta=0.2, AR=1.1)` the incoming partition sits near
0.21 with variance 0.028, and the unconstrained fit put the invariant mean at
0.79. That is not arithmetic error; it is a question the data cannot answer,
being asked anyway.

## 3. What the bridge does

Make the kernel reversible with respect to `Beta(2,2)` by construction:

```
p(z' | z, eps)  ∝  6 z'(1-z') · h(z) h(z') · exp( λ₃ z z'
                     + λ₁ z' + λ₂ z'² + λ₄ eps z' )
```

`h` is the symmetric Sinkhorn potential of the coupling `exp(λ₃ z z')`, found
by iterative proportional fitting against the `Beta(2,2)` marginal. With the
tilt `(λ₁, λ₂, λ₄)` held at zero, `π(z) p(z'|z) = π(z') p(z|z')` exactly, so
`Beta(2,2)` is the invariant law **for any memory `λ₃`** — to machine
precision, with no fitting involved. Only the dissipative tilt, which is
fitted, can move the fixed point. Elastically there is no dissipation, so the
tilt is held at zero and equipartition becomes an identity of the kernel.

`λ₃` remains a free, fitted parameter. This is the point: memory strength and
the equilibrium it relaxes towards are *decoupled*. The old form entangled
them, which is why tightening the memory dragged the invariant law off 1/2.

Two exact structural identities fall out, and both are pinned by tests:
`log h(0) - log h(1) = λ₃/2`, and `h ≡ 0` when `λ₃ = 0`.

## 4. Verification on the real 200k sentinel

Held-out 4:1 split on the actual shards under the **production kinematic
weight** (128 offsets), bridge versus the unconstrained conditional tilt.
Log-density is per event, in nats; higher is better. All nodes elastic.

| node | bridge ll | free ll | gap | bridge inv. mean | free inv. mean |
|---|---|---|---|---|---|
| `AR=1.1, th=2` | **1.2445** | 1.2421 | −0.0024 | **0.500000** | 0.487564 |
| `AR=2, th=0.2` | 0.1568 | 0.1700 | +0.0132 | **0.500000** | 0.504315 |
| `AR=2, th=1` | 0.1679 | 0.1683 | +0.0004 | **0.500000** | 0.489089 |
| `AR=2, th=2` | 0.1581 | 0.1587 | +0.0006 | **0.500000** | 0.490972 |
| `AR=3, th=0.2` | 0.1324 | 0.1451 | +0.0127 | **0.500000** | 0.492681 |
| `AR=3, th=1` | 0.1523 | 0.1542 | +0.0019 | **0.500000** | 0.484952 |
| `AR=3, th=2` | 0.1376 | 0.1400 | +0.0024 | **0.500000** | 0.484027 |

Three readings:

1. **The free fit still misses equipartition by 1 to 3 percent** under the
   correct weighting — 0.4840 to 0.5043. Fixing the incoming-weight artifact
   did not fix this, so the case for the bridge stands on its own.
2. **The constraint is close to free.** The largest held-out cost is 0.0132
   nats, well inside the 0.02 model-form tolerance, and it occurs at the two
   `theta = 0.2` nodes where the incoming partition is far from equilibrium.
   At `AR = 1.1` the bridge is *better*, with one parameter instead of three.
3. **Elastically the invariant law is exactly 0.500000** at every node, by
   construction.

An earlier draft of this table was computed with the static shadow weight and
reported different free-fit invariant means (0.4982, 0.7936, 0.4817, 0.4883).
Those are superseded by the numbers above. The qualitative conclusions did not
change; the numbers did.

## 5. The honest caveat

Imposing equipartition converts `theta*(alpha = 1) = 1` from a **validation**
into an **identity**. It is no longer evidence that the pipeline is correct,
and it must not be presented as such.

The honest accuracy statement for the elastic limit remains the *unconstrained*
fit's departure — `theta* = 1.012 / 0.946 / 0.918`, i.e. 1 to 8 percent. That
is the measured error bar of the pipeline and it has not improved. What has
changed is that the **deployed** kernel can no longer violate a law we know
exactly. Those are different claims and both belong in the write-up.

The sharpest statement of the risk is a test. On synthetic data whose
invariant partition is `Beta(3,4)` — mean `3/7`, not `1/2` — the conditional
form recovers `3/7` and the bridge reports `0.500000`. The bridge cannot
detect a wrong reference measure; it will impose it silently. Our grounds for
believing `Beta(2,2)` is right for this system are independent of the fit: the
generator draws incoming states from `z(1-z)(z/theta + 1 - z)^-4`, and the
measured elastic node at `theta=1, AR=1.1` has mean 0.4990 and variance 0.0502
against `Beta(2,2)`'s 0.5 and 0.05. That is good evidence, but it is evidence,
not proof, and it is now pinned by
`test_bridge_imposes_its_reference_law_even_when_the_data_disagree`.

The residual an earlier draft claimed here — a 4 percent incoming-partition
bias at `AR = 3` — does not exist; see the correction in section 2.

## 6. Relation to Hong & Morris (2022)

Their DSMC for spherocylinders draws the post-collisional partition from an
`AR`-independent polynomial, their Eq. (13). Peak-normalised against
`Beta(2,2)`:

| `z` | Eq. (13) | `Beta(2,2)` | rel. |
|---|---|---|---|
| 0.10 | 0.3644 | 0.3600 | +1% |
| 0.25 | 0.7469 | 0.7500 | −0% |
| 0.50 | 0.9992 | 1.0000 | −0% |
| 0.75 | 0.8036 | 0.7500 | **+7%** |
| 0.90 | 0.4618 | 0.3600 | **+28%** |
| 0.98 | 0.2032 | 0.0784 | **+159%** |

It matches `Beta(2,2)` to about 1 percent through the bulk `z <~ 0.5`, which is
independent evidence that `Beta(2,2)` is the right reference for this system.
It departs badly in the upper tail, because an unconstrained quartic cannot
vanish at `z = 1` where a `Beta(2,2)` must — Eq. (13) evaluates to 0.00491
there. That tail surplus is exactly where the mean bias below comes from.

(An earlier draft of this report quoted only `z = 0.1, 0.25, 0.5` and claimed
1 percent agreement. Those are precisely the three points where it agrees; the
tail was omitted. Corrected here.)

Two differences matter:

- **Their fit has the same defect, unconstrained.** Normalised as a density,
  Eq. (13) has mean 0.5145 and second moment 0.3182, against 0.5 and 0.3. That
  is a 2.9 percent departure from equipartition in the published reference
  distribution — the same order as the 1 to 3 percent we were seeing, and for
  the same reason: a four-parameter unconstrained polynomial fit with nothing
  holding it to the equilibrium it is supposed to relax to.
- **Memory.** Their kernel is memoryless: an exchange either happens with
  probability `1/Z_R` and redraws from Eq. (13), or nothing happens. All the
  `AR` dependence lives in `Z_R`, which they find to be `5/3` for every aspect
  ratio once time is measured in collisions per particle. Our `λ₃` is the
  continuous generalisation of that single knob, fitted per node, with the
  equilibrium held fixed independently.

Their Fig. 8 supports the physical reading directly. In `t`, higher aspect
ratios relax faster; in `τ` (collisions per particle) all aspect ratios
collapse onto one curve. The geometry sets the *collision rate*, not the
per-collision redistribution. That is exactly the separation the bridge encodes
structurally: `λ₃` (per-collision memory) is free and node-dependent, while the
equilibrium it relaxes towards is fixed and universal.

## 6a. The replacement validation: `Z_R` and the spectrum

Losing `theta*(alpha=1)` as evidence does not leave the bridge unvalidated,
because the bridge *separates* the equilibrium (now imposed) from the
relaxation rate (still entirely free) — and Hong & Morris measured the
relaxation rate independently.

Reversibility makes the kernel self-adjoint in `L2(Beta(2,2))`, so its spectrum
is real. Verified on the deployed implementation: `max|Im mu| < 4e-17` at every
memory tested. Two consequences the old kernel could not offer:

- **Relaxation cannot ring.** `theta(t)` is a sum of decaying exponentials with
  no oscillatory modes; it cannot overshoot or spiral.
- **An H-theorem.** From a deliberately distorted start at `lambda3 = 5`, the
  KL divergence to `Beta(2,2)` falls monotonically —
  8.10e-1, 3.08e-2, 1.73e-3, 9.73e-5, 5.49e-6, 3.10e-7, 1.75e-8, 9.85e-10 —
  about a factor 18 per collision. The elastic kernel provably relaxes to
  equipartition and can do nothing else.

The second eigenvalue `mu1` governs the decay and its eigenfunction is very
nearly `z` itself, so Jeans' equation is *derived* rather than fitted, with
`Z_R = -1/ln(mu1)`. Hong & Morris measured `Z_R = 5/3`. On our grid:

```
Z_R = 5/3  <=>  mu1 = e^-3/5 = 0.548812  <=>  lambda3 = 15.1650
```

This is a genuine, free, published check on a parameter the bridge leaves
completely free. **Measured on the elastic sentinel nodes** (production
kinematic weight, 128 offsets):

| node | `lambda3` | `mu1` | `Z_R` | `2 Z_R` |
|---|---|---|---|---|
| `AR=1.1, th=2` | 180.26 | 0.9500 | 19.49 | — |
| `AR=2, th=0.2` | 7.34 | 0.3312 | 0.905 | 1.81 |
| `AR=2, th=1` | 6.35 | 0.2934 | 0.815 | 1.63 |
| `AR=2, th=2` | 5.81 | 0.2716 | 0.767 | 1.54 |
| `AR=3, th=0.2` | 6.24 | 0.2890 | 0.806 | 1.61 |
| `AR=3, th=1` | 5.32 | 0.2512 | 0.724 | 1.45 |
| `AR=3, th=2` | 4.58 | 0.2191 | 0.659 | 1.32 |

Three things follow.

**The `AR = 1.1` value is not a failure of AR-independence — it is a required
limit.** Smooth spheres cannot exchange energy between modes at all, so
`Z_R -> infinity` as `AR -> 1`. We measure `Z_R = 19.5` at `AR = 1.1`. Hong &
Morris's collapse is claimed for `AR = 2..5` only; `AR = 1.1` lies outside it,
and the closure reproduces the correct singular behaviour there without being
told to.

**Within their range the collapse holds.** `Z_R` spans 0.66 to 0.91 across
`AR = 2` and `3` and across three temperature ratios — roughly AR-independent,
as Fig. 8b requires.

**The absolute value matches after one physical factor.** Only the *relative*
translational energy is redistributed in a collision; the centre-of-mass half
is untouched, so the gas relaxes about twice as slowly as the pair chain.
`2 Z_R = 1.32` to `1.81`, mean 1.56, against the published `5/3 = 1.667` —
within 7 percent.

> **State this carefully.** The factor of 2 is physically motivated but was
> applied *after* seeing the numbers, so it is a plausible reconciliation, not
> yet a validation. Test 3 above — run the 0-D DSMC at `alpha = 1` from
> `T_tr = 1, T_rot = 0` and overlay `dE_k(tau)` on their Fig. 8b — compares
> like with like and settles it without any such factor. Do that before
> claiming the agreement.

There is also a clear **`theta` dependence** at fixed `AR`: `lambda3` runs
7.34 / 6.35 / 5.81 at `AR = 2` for `theta = 0.2 / 1 / 2`, a 26 percent spread.
A memoryless single-`Z_R` model cannot represent that. It is a concrete
statement of what this closure adds over Hong & Morris.

## 6b. Why `Beta(2,2)`, from counting

At fixed pool energy the phase-space volume with translational share `z` goes
as `z^(zeta_t/2 - 1) (1-z)^(zeta_r/2 - 1)`. Here `zeta_r = 4` (two transverse
spins on each of two rods) and `zeta_t = 5 - 2*omega = 4` — three relative
translational degrees of freedom, promoted to four by the collision-flux
weighting proportional to `g`. Hence `Beta(2,2)`. This is why the naive
"3 translational versus 4 rotational" count, which would give `Beta(3/2,2)` and
a mean of `3/7`, is the wrong one: it omits the flux weighting.

## 7. Physical reading

Equipartition is not something a single collision does. It is what the
*ensemble* is driven to, because at fixed total energy the `Beta(2,2)` partition
is the one with the most phase-space volume behind it. The dynamical statement
of "there is a restoring force holding the system there" is precisely detailed
balance: `π(z) p(z'|z) = π(z') p(z|z')` says that whatever the kernel does in
one direction it undoes in proportion in the other, so `π` cannot drift. There
is no extra force to add — reversibility *is* the restoring mechanism, and the
Sinkhorn potential `h` is the minimal deformation that installs it without
touching the memory.

Letting a fitted kernel float and hoping its invariant law lands on `Beta(2,2)`
asks the data to re-derive a symmetry we already know, using collisions that
individually carry very little information about it — worst exactly where the
coupling is weak (`AR=1.1`) or the incoming states are far from equilibrium
(`theta=0.2`). Those are the nodes that failed. Holding the reference state and
fitting only the departure from it puts the burden on the data only where the
data is actually informative.

## 8. Cost

The profile likelihood over `λ₃` needs ~16 inner solves cold, ~9 warm. The
inner Newton originally carried a fixed `1e-12` ridge, which leaves the step
enormous when the model covariance is near singular; each iteration then paid
for ~35 backtracking halvings over the full event set. One node spent 2103 of
its 2220 seconds inside a *single* such solve. Levenberg-Marquardt damping
replaced it: on that node the cold fit went from **2220.2 s to 182.3 s** (12.2x)
and the worst single inner solve from **2102.7 s to 39.6 s** (53x), with the
same 16 profile evaluations and an identical `lambda3 = 7.2517`. The estimate
does not move; only its conditioning does.

Bootstrap replicates and the held-out model-form refit search a multiplicative
bracket `WARM_BRACKET = (0.4, 2.5)` around the point estimate from a converged
tilt, rather than all nine decades. `λ₃` stays genuinely free, so its bootstrap
spread is real rather than collapsed to zero: on the `alpha=0.8, AR=2` node the
cold estimate is 7.2517 and warm resamples give 7.2554 / 7.2284 / 7.1947.
