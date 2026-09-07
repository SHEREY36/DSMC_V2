# Handoff: the closure is validated, the DSMC coupling is not

> **Superseded on 2026-09-07:** the coupling defect was isolated and repaired.
> See `reports/dsmc_coupling_resolution.md`.  Statements below describe the
> pre-repair code and are retained as diagnostic history.

**Branch** `closure-v2-repairs` · **Suite** 151 passing · **Written** 2026-09-07

This is a handoff to whoever picks up the DSMC coupling problem. It says what
is established, what is broken, what was tried, and — at some length — which
diagnostics are misleading, because most of the wasted effort in this work came
from measuring the wrong thing convincingly rather than from hard physics.

---

## 1. One-paragraph status

The closure reproduces the DEM steady temperature ratio to better than 1 per
cent at AR = 2 and 3, and is exact in the elastic limit by construction. The
DSMC that consumes it does not: it shifts `theta*` by **+4.8 per cent
elastically and −12.8 per cent inelastically at the same aspect ratio**. Those
signs are opposite, so it is not one stray scale factor. Everything upstream of
the runtime — generator, measure, kernel, anchor, gates — is in good shape.

---

## 2. What is established

Against MFIX DEM (`kT_1_kTr_1`, phi = 0.01, 64^3, N = 2003):

| AR | alpha | closure | DEM | error |
|---|---|---|---|---|
| 2.0 | 0.95 | 0.9867 | 0.9792 | +0.8% |
| 2.0 | 0.80 | 0.932 | 0.9365 | −0.5% |
| 3.0 | 0.95 | 0.9923 | 0.9962 | −0.4% |
| 3.0 | 0.80 | 0.960 | 1.0158 | −5.5% |
| any | 1.00 | 1.0000 | 1.0000 | exact, every theta, every AR |

No flow calibration anywhere: the kernel is fitted only to isolated pair
collisions. The elastic row is the ruler — an elastic gas *must* sit at
`theta = 1`, so any deviation there is pure error with no physics in it. Use it
constantly.

Independent checks that also pass: the spectrum is real (`max|Im mu| < 4e-17`),
an H-theorem holds, and the second eigenvalue reproduces Hong & Morris's
`Z_R = 5/3` including the smooth-sphere divergence at AR -> 1, which nothing
told it about.

---

## 3. The open problem

Closure versus runtime, same nodes, same artifact:

| AR | alpha | closure | DSMC | truth | runtime gap |
|---|---|---|---|---|---|
| 2.0 | 1.00 | 1.0000 | 1.0475 | 1.0000 | **+4.8%** |
| 2.0 | 0.95 | 0.9867 | 0.8592 | 0.9792 | **−12.8%** |
| 3.0 | 0.95 | 0.9923 | 0.9466 | 0.9962 | **−4.6%** |

Reproduce with `models/microscopic_closure_v2/closure_v2.npz` (24 nodes, AR 2
and 3), phi = 0.01, 64^3 box, `t_end = 60`, `invariant_corrections: false`.

**Opposite signs at one aspect ratio rule out a single mis-scaling.** A wrong
normalisation moves everything the same way. Something interacts with the
dissipative branch specifically.

### Candidates, in the order I would test them

1. **Pair selection.** The kernel is fitted on the CTC *collision* ensemble,
   whose incoming partition is 0.4860 at AR = 2 and 0.4803 at AR = 3 — not 0.5,
   because collisions favour fast rotators. The runtime is supposed to
   reproduce that with a second acceptance stage on `A_perp * g(Xi)`. **The
   decisive measurement is to instrument one alpha = 1 run and record the
   `<z_in>` its own collisions actually carry, then compare with 0.4860.** That
   separates "selects different pairs" from "feeds the kernel different
   arguments" in a single run. Do this first; it is one short run.

2. **The loss handed to the kernel.** `a = lambda1 + lambda3 z + lambda4 eps`.
   `lambda4` reaches 352 at some nodes, so the kernel is extremely sensitive to
   `eps`. The runtime draws `gamma` from the frozen v1 BL tables, whose mean is
   13–28 per cent below the CTC's, and rescales only the *mean*
   (`artifact.py`, `sample_energy`). Because the kernel is nonlinear in the
   loss, `E[K(z'|z,gamma)] != K(z'|z,E[gamma])`. This is dissipative-branch-only,
   which fits the sign pattern. Check per-collision, not in the mean.

3. **Aspect-ratio interpolation.** The artifact now has two AR values, and a
   query at non-grid theta triangulates in 3-D across them. Earlier, an
   AR = 3-only artifact (with AR replicated at 2.99/3.00/3.01) gave +1.4 per
   cent on the elastic ruler; the two-AR production artifact gives +4.8. That
   is suggestive. The seven-aspect-ratio campaign now running on Negishi
   (AR 1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0) may reduce this on its own —
   re-measure before chasing it.

---

## 4. Diagnostics that will mislead you

Read this section before measuring anything. Each of these produced a
confident, wrong conclusion during this work.

### 4.1 `outcome_weights(run)` without a propensity

Returns the **deprecated static shadow weight** `1/A_perp`, not the production
weight. Its own docstring says it misses the rotational enhancement. I used it
for a quick comparison, then quoted `<z_in> = 0.4807` from it as if it were
production. An entire external review then built its top-ranked recommendation
(a "quadrant sampling bug") on that number. Under the real kinematic propensity
the value is **0.4996**. There was no bug.

**Always pass `propensity=_run_propensity(run, offsets)` explicitly.**

### 4.2 Using the kernel's invariant law as the z <-> theta map

After the anchor fix, the elastic kernel's invariant is the equilibrium law at
*every* theta — `0.4990, 0.4990, 0.4990`. That is correct and is the whole
point of the fix. It also makes it **degenerate as a calibration**. I used it
twice as a `theta` map and got nonsense both times (everything clamping to
0.200, then `theta* = 2.000` for an elastic node).

**The z <-> theta calibration comes from `incoming_law` on the elastic nodes**,
which is theta-dependent. It is in the node JSON. The artifact does not carry
it.

### 4.3 `theta = z/(1-z)`

Wrong. It assumes the proposal ensemble and returns **0.9224 where the physics
demands exactly 1.000**. That single bad map manufactured an apparent 8 per
cent "closure bias" that sent the work down a false path for hours. Calibrate
against the measured incoming law instead.

### 4.4 `sacct | grep -v COMPLETED`

Filters out the successes, leaving only `RUNNING` rows — which reads as
"everything is stuck". Nothing had failed. Check `State`, `ExitCode`, and
non-empty `.err` files.

### 4.5 CTC tasks finishing in under a minute

Correct behaviour, not a crash. Conservative advancement made generation ~5800x
faster: a 200k-hit row is ~16 s on 20 threads, so a strided array task doing
three rows completes in ~50 s.

### 4.6 A background build with no output

`build_artifact` spends 10–20 minutes per *uncached* kinematic propensity before
printing anything. I read an empty log as a crash twice. Check
`ps -o etimes,pcpu` — 100 per cent CPU means it is working.

### 4.7 `pkill -f <pattern>`

If the pattern appears in your own command line, pkill matches the invoking
shell and kills it, losing any un-written edits. Cost me one applied patch.
Exclude `$$` and `$PPID`, or match the interpreter precisely.

---

## 5. Errors made in the code, and why they mattered

Recorded because each has a recognisable shape that will recur.

**Anchoring each node on its own incoming law.** The Sinkhorn bridge is
reversible with respect to a reference measure. I measured that reference from
each node's own partition, which is only the equilibrium at `theta = 1`.
Everywhere else it declared an off-equilibrium law stationary and **removed the
restoring force entirely** — the refitted grid gave `theta* = 0.200, 1.000,
2.000` at alpha = 1, an elastic kernel that preserves whatever it is handed.
The reference is a property of aspect ratio, injected, never measured per node.
Caught by the alpha = 1 ruler.

**Claiming the collision clock was frozen "by construction".** I normalised
`g(Xi)` to mean one and asserted the rate could not move. That holds only on
the sample the curve was fitted on: the `Xi` distribution shifts with theta, and
the true multiplier `<A g>/<A>` runs **1.1527 at theta = 0.2 and 0.9519 at
theta = 2** for AR = 3. Fifteen per cent, asserted to the user as impossible.
Each curve is now divided by its own multiplier.

**Adding a static-area acceptance stage without measuring it.** `A_perp`
depends on orientation but not spin magnitude, while the partition depends on
spin — so it reproduced the proposal ensemble to four decimals and did
**nothing**. One five-minute measurement before writing it would have shown
that. The rotation-number surrogate `A_perp * g(Xi)` matches the exact
propensity to four decimals.

**A shadowed variable.** The warm-start bracket centre was also named `anchor`
and overwrote the reference law, so warm-started refits built the Sinkhorn
potential against a float. It raised, and was found. With compatible types it
would have silently fitted the wrong kernel.

**Two silent defaults.** The stability gate defaulted the incoming law to
Beta(2,2) when absent — theta-independent, which erases exactly what the gate
measures. The runtime accepted every pair when the enhancement table was
missing, reverting to the proposal ensemble with no error. Both now raise.
**Prefer a hard failure to a plausible default anywhere in this pipeline.**

---

## 6. Architecture, briefly

```
HS_CTC_v2/            Fortran generator. Untouched, validated, no regeneration
                      needed. Raw hits preserve the elastic invariant to 4e-4.
  -> shards           results/ctc_closure_200k/<stage>/alpha_..._AR_..._shard_00

Coll_Models_v2/
  estimate.py         estimate_node(): weights, fits, gates. MEASURE="collision"
                      (raw hits). Writes cell_features, incoming_law,
                      incoming_law_energy, anchor_c1/c2.
  fit_exchange.py     the Sinkhorn bridge; measure_anchor()
  projections.py      bridge_potential/logpdf/stationary/mean_map,
                      energy_quantile_table
  excitation.py       reweighted first-order excitation (no CTC needed)
  artifact.py         build_artifact(): per-node g(Xi), stability gate,
                      closure_v2.npz
  -> artifact         models/microscopic_closure_v2/closure_v2.npz (schema 2.3.0)

DSMC_0D_v2/
  artifact.py         VariationalClosure: interpolation, sample_energy
  kernel.py           collide(), accept_orientation()      <-- problem lives here
  simulation.py       NTC loop
```

### Facts worth having at hand

* Reference measure is **not** Beta(2,2). It is the collision-ensemble
  equilibrium: 0.4990 (AR 1.1), 0.4860 (AR 2), 0.4803 (AR 3).
* Fit on **raw hits**. `1/P` reweighting breaks the elastic invariant by
  −0.0117 at AR = 3, scaling with aspect ratio; raw hits hold it to −0.0004.
* `lambda3` is a bad interpolation coordinate — 171 at AR 1.1 against 6.2 at
  AR 2. A bounded exchange strength (`kappa = 1 - rho_z`, i.e. `p_exch`) is far
  better behaved: 0.056 / 0.710 / 0.760. But **do not impose a (AR-1)^2 torque
  law**: measured `p_exch/(AR-1)^2` is 5.55, 0.710, 0.19 across AR 1.1, 2, 3 —
  a thirty-fold variation. Fit the exponent on the seven aspect ratios now
  being generated.
* `W2` is **structurally unidentifiable** for smooth spherocylinders: axial
  spin measures 3.9e-13 because a smooth rod cannot spin about its own axis.
  The 14-column design has 13 usable columns for this particle model.
* `PiPi`, `QQ`, `RtRt` are non-negative contractions, so excitation moves them
  in one direction only. That is the invariant's property, not a defect.
* AR = 1.1 inelastic fixed points sit **below theta = 0.2**, outside the
  sampled grid; the stability gate refuses those nodes, correctly. The
  `ar_low_theta` campaign (theta 0.0125 .. 0.15) addresses it.

---

## 7. What to do next

1. **Instrument one alpha = 1 run** and compare the DSMC's own collision-sampled
   `<z_in>` against the CTC's 0.4860 at AR = 2. One run, decisive.
2. If selection matches, check the **per-collision** loss the kernel receives
   against the `eps` it was conditioned on. Do not compare means.
3. Re-measure after the seven-aspect-ratio campaign lands; some of the
   interpolation error should go on its own.
4. Only then USF. It adds shear on top of a coupling that already mis-tracks
   HCS by 5 to 13 per cent.
5. The excitation module is implemented and unit-tested but **not wired into
   node estimation**. Four families reach usable amplitudes at ESS >= 0.83;
   `A_cu` and `qtr2` have weak reach (~1e-3) and need better scores.

**Judge every change against the alpha = 1 ruler before anything else.** It has
a known exact answer, it has no free parameters, and every real error in this
work showed up there first.
