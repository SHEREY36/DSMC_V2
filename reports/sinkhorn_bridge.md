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

Two measurements make the diagnosis concrete. For each sentinel node, the
weighted mean of the *incoming* partition and of the *outgoing* partition:

| node | mean `z_in` | mean `z_out` | var `z_in` |
|---|---|---|---|
| `alpha=1, theta=1, AR=1.1` | 0.4990 | 0.4990 | 0.0502 |
| `alpha=1, theta=1, AR=3.0` | 0.4807 | 0.4810 | 0.0512 |
| `alpha=1, theta=0.2, AR=1.1` | 0.2120 | 0.2281 | 0.0281 |

`Beta(2,2)` has mean 0.5000 and variance 0.0500. The first row matches it to
three decimals, which confirms that `Beta(2,2)` is the right reference for this
system — as it must be, since the generator draws incoming states from
`incoming_partition_density(theta, z) = z(1-z) (z/theta + 1 - z)^-4`, whose
`-4` exponent encodes the four-versus-four degree-of-freedom structure.

The second row is the diagnosis. `mean z_out` tracks `mean z_in` to four
decimals (0.4810 vs 0.4807). The fitted kernel was *correctly* stationary with
respect to the sample it was shown — and that sample was itself 4 percent off
`Beta(2,2)`. The 0.4817 invariant mean was never a defect in the kernel. It
was the estimator faithfully reporting the invariant law of a kernel fitted to
a biased incoming sample.

The third row is the extreme case. The incoming partition is concentrated near
0.21 with variance 0.0281. Asking where *that* kernel would drive the system
after infinitely many collisions is an extrapolation far outside the support of
the data, and the answer (0.7936) is meaningless — not wrong arithmetic, just a
question the data cannot answer.

So the invariant law was a **derived** quantity, sensitive to sampling bias in
a covariate, standing in for a law we already knew exactly. That is the wrong
division of labour.

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

Held-out 4:1 split on the actual shards, bridge versus the unconstrained
conditional tilt. Log-density is per-event, in nats; higher is better.

| node | bridge | free tilt | gap | bridge invariant mean | free invariant mean |
|---|---|---|---|---|---|
| `a=1, th=1, AR=1.1` | **1.2064** | 1.2053 | −0.0011 | **0.500000** | 0.498166 |
| `a=1, th=0.2, AR=1.1` | **1.4010** | 1.3958 | −0.0053 | **0.500000** | 0.793584 |
| `a=1, th=1, AR=3.0` | 0.1499 | 0.1527 | +0.0029 | **0.500000** | 0.481709 |
| `a=1, th=2, AR=2.0` | 0.1559 | 0.1569 | +0.0010 | **0.500000** | 0.488266 |
| `a=0.5, th=1, AR=3.0` | 0.2038 | 0.2038 | −0.0001 | 0.366373 | 0.366463 |
| `a=0.8, th=1, AR=2.0` | 0.2055 | 0.2055 | −0.0000 | 0.462104 | 0.462158 |

Three readings:

1. **Elastically the invariant law is exactly 0.500000** at every node — by
   construction, not by fit.
2. **The constraint is free.** At `AR=1.1` the bridge is *better* held-out,
   with one parameter instead of three. At `AR=2` and `3` the free fit wins by
   at most 0.003 nats, negligible against removing a 3 percent systematic. At
   the 48-sigma node the bridge both fixes the invariant law *and* fits better,
   which is the signature of a badly identified parameter in the free form.
3. **It is inert where it should be.** Inelastically the two agree to 0.0001
   nats and their invariant means agree to four decimals. The bridge does not
   drag dissipative fixed points toward 1/2. Had it done so, the idea would
   have been wrong.

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

A concrete residual, surfaced by the table in section 2: the incoming-partition
sampling at `AR=3` is 4 percent off its `Beta(2,2)` target. That is a
generator/weighting issue, not a kernel issue, and the bridge conceals it
rather than fixing it. The `incoming_partition_pass` gate is what watches it.

## 6. Relation to Hong & Morris (2022)

Their DSMC for spherocylinders draws the post-collisional partition from an
`AR`-independent polynomial, their Eq. (13). That polynomial is very nearly
`Beta(2,2)`: peak-normalised it agrees to about 1 percent at `z = 0.1, 0.25,
0.5` (0.3644 vs 0.36, 0.7272 vs 0.7296, 0.9993 vs 1.0). So the reference
measure the bridge deforms is not an arbitrary modelling choice — it is
essentially what their DEM ensembles measured.

Two differences matter:

- **Their fit has the same defect, unconstrained.** Normalised as a density,
  Eq. (13) has mean 0.5128 and second moment 0.3153, against 0.5 and 0.3. That
  is a 2.6 percent departure from equipartition in the published reference
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
