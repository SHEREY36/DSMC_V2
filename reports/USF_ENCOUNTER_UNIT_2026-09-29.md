# USF closure repair: the DSMC event unit — 2026-09-29

## Result in one paragraph

The rod-USF errors (temperature ratio up to 17 % low at small restitution,
`T*` up to 27 % low at large restitution, `|P*_xy|` 5–12 % low, `N1` 15–30 %
low) are not missing flow-invariant physics.  They are an **event-unit
inconsistency** in the DSMC collision operator: every collision law fitted from
CTC describes one complete *encounter* (first contact to final separation,
`<k>` = 1.0–1.31 contacts), but the runtime fired those laws at the frozen
v1 rate `sigma_c`, which is a *contact* rate, and removed the per-*contact* BL
loss.  A smaller, genuine model-form gap remains in the angular law, which
ignores the incoming energy partition.  Both were measured directly by applying
exact CTC and the DSMC operator to the *same* incoming USF pairs.  The repair
needs no new CTC data and no DEM information.

## 1. Exact balance that fixes the USF steady state

For homogeneous USF (`u_x = gdot y`) the kinetic stress and rotational energy
obey, without approximation,

```
dP_ij/dt = -gdot (d_ix P_jy + d_jx P_iy) + Lambda_ij[f],     dE_rot/dt = Lambda_rot[f],
Lambda_ij[f] = (n^2/2) Int dGamma f1 f2 g sigma(Gamma) < (m/2) Delta(g_i g_j) >_Gamma .
```

The steady state is therefore set entirely by *rate x per-event production*.
Two combinations dominate (near-elastic scaling, verified below):

```
T*  ~ 1 / (s_eta s_zeta),        |P*_xy| ~ sqrt(s_zeta / s_eta),
```

where `s_eta` and `s_zeta` are the model/true ratios of the shear-stress
relaxation and dissipation production rates.  USF also imposes
`Lambda_rot = 0` on its own, while HCS only fixes the ratio of the modal
cooling rates; the rotational drain enters USF at full weight but nearly
cancels in the HCS drift.  This is why a correct HCS temperature ratio never
implied a correct USF temperature ratio.

## 2. Measurement on identical pairs

Codex's 168 exact CTC replays of DSMC-USF post-NTC incoming pairs
(`outputs/usf_nonlinear_v2_20260928_ctc/replay`) give the true productions on
the DSMC's own states.  Per flux sample, CTC production is
`(A_prop/n_attempts) sum_hits Delta X`; DSMC production is
`sigma_c A_perp g(Xi)/A_mean * E[Delta X | x]` using the runtime samplers.

| AR | DSMC events / CTC encounters | DSMC events / CTC contacts | BL mean / CTC encounter mean loss |
|---:|---:|---:|---:|
| 1.1 | 1.02 | 1.02 | 0.86–0.97 |
| 1.5 | 1.08 | 0.96–1.03 | 0.86–0.92 |
| 2.0 | 1.14 | 0.92–1.02 | 0.78–0.82 |
| 3.0 | 1.23 | 0.93–1.05 | 0.72–0.88 |

The encounter cross-section is kinematic: at every node it is identical across
`alpha` to four figures.  Legacy xy-stress relaxation was 1.01x (AR 1.1) to
1.36x (AR 3) too fast.  Independently, inverting the DEM steady state with the
scaling above gives `(s_eta, s_zeta) = (1.28, 1.07)` at AR 3, alpha 0.95; the
direct probe gives `(1.27, 1.08)`.

Per encounter at AR 3, alpha 0.5: CTC rotational change `-0.0324 E`; legacy
DSMC `-0.0030 E`; DSMC with the encounter loss `-0.0326 E`.  The conditional
energy law `<z'|z>` agrees with CTC in every z bin — the energy kernel is not
the problem.  The angular law is: CTC `<P2(ghat.ghat')>` rises from 0.006 to
0.22 across incoming z, whereas the artifact law is flat (depends on z' only),
so the energy-weighted stress destroyed per event is 3–10 % too large.

## 3. Repair

1. **Encounter clock.** `sigma = sigma_c(AR) r(theta, AR)`,
   `r = sigma_enc/sigma_c` from CTC hits/attempts
   (`DSMC_0D_v2/models/encounter_cross_section_v1.json`; 0.77–1.04).
2. **Encounter loss.** The BL Beta draw is rescaled to the CTC per-encounter
   mean `eps_enc` that the energy kernel was fitted with; the routing
   covariate `lambda4 eps` is then the removed loss itself.
3. **Angular memory.** Energy-tilted conditional I-projection
   `p(c|z,z') ~ exp[(eta1 + xi z' + rho1 z) c + (eta2 + zeta z' + rho2 z) P2(c)]`,
   fitted with weight `E_f` (post-collision pair energy) and a weak
   `N(0, 5^2)` prior on the memory coefficients
   (`Coll_Models_v2/scripts/fit_angular_memory_table.py`,
   `DSMC_0D_v2/models/angular_memory_v1.json`, all 144 artifact nodes).
   Because `z' P2(c)` is a sufficient statistic, the fit reproduces
   `E[E_f z' P2]`, the traceless-stress production; the `z` terms carry it off
   equilibrium.  Mixture sampling on the energy kernel's stencil, with a
   per-node natural-parameter support gate.

Runtime opt-ins: `microscopic_closure.event_unit: encounter` and
`microscopic_closure.angular_memory: <table>`.  Defaults are unchanged.
DSMC collision counts are then encounters; multiply by `<k>` (stored in the
encounter table) before comparing with DEM contacts.

## 4. Verification

Operator level, USF replays, DSMC/CTC production (legacy -> repaired):

| AR | alpha | xy relaxation | dissipation | rotational residual / loss |
|---:|---:|---|---|---|
| 1.5 | 0.50 | 1.155 -> 1.025 | 0.916 -> 0.980 | 0.066 -> -0.009 |
| 2.0 | 0.50 | 1.228 -> 1.030 | 0.882 -> 0.997 | 0.136 -> 0.011 |
| 2.5 | 0.50 | 1.299 -> 1.008 | 0.870 -> 1.001 | 0.142 -> -0.014 |
| 3.0 | 0.50 | 1.360 -> 1.030 | 0.896 -> 1.011 | 0.161 -> 0.004 |
| 3.0 | 0.80 | 1.332 -> 1.017 | 1.032 -> 1.000 | 0.078 -> -0.010 |
| 3.0 | 0.95 | 1.285 -> 1.036 | 1.089 -> 1.005 | 0.073 -> 0.000 |

The angular law fitted on the theta=1 HCS node reproduces the USF stress
destruction per encounter to 0–1 % out of sample (production law: 2–6 %).

HCS: `theta*` is invariant to the clock and the angular law does not act on a
homogeneous isotropic state; the encounter loss moves the drift roots at the
DEM benchmark nodes by at most about 2 %.

Local USF, encounter unit only (repairs 1–2), N = 10,000, one cold trajectory
per case with the usf-crossflow-v1 seed, shear rate, dt and tau window
[96, 160].  Error relative to DEM (legacy study -> encounter unit):

| AR | alpha | theta | T* | P*_xy | N1 |
|---:|---:|---|---|---|---|
| 1.5 | 0.5 | -13.6 -> +2.2 % | +5.1 -> +5.7 % | -5.0 -> -1.7 % | -18.6 -> -7.8 % |
| 1.5 | 0.8 | -3.9 -> -0.5 % | -5.0 -> -0.6 % | -7.1 -> -3.1 % | -14.0 -> -4.7 % |
| 2.0 | 0.5 | -16.9 -> -1.8 % | +0.8 -> -0.8 % | -6.6 -> -1.3 % | -24.1 -> -7.2 % |
| 2.0 | 0.8 | -5.5 -> -1.3 % | -3.3 -> -2.2 % | -11.6 -> -2.6 % | -24.9 -> -7.0 % |
| 3.0 | 0.5 | -16.4 -> -1.9 % | -9.2 -> -4.4 % | -8.5 -> -1.9 % | -30.3 -> -9.1 % |
| 3.0 | 0.8 | -3.8 -> -0.3 % | -24.8 -> -5.1 % | -10.4 -> -3.9 % | -20.3 -> -5.4 % |

alpha = 0.95, window [216, 360], encounter unit only:

| AR | theta | T* | P*_xy | N1 |
|---:|---|---|---|---|
| 1.5 | -0.8 -> -0.1 % | -8.7 -> -2.2 % | -5.7 -> -2.3 % | -9.8 -> -5.4 % |
| 2.0 | -1.6 -> -0.6 % | -7.3 -> -3.7 % | -11.7 -> -2.7 % | -18.0 -> +0.6 % |
| 3.0 | -0.5 -> -0.1 % | -26.9 -> -4.9 % | -8.7 -> -2.3 % | -17.3 -> -6.5 % |

Full model (repairs 1–3), same protocol:

| AR | alpha | theta | T* | P*_xy | N1 |
|---:|---:|---|---|---|---|
| 2.0 | 0.5 | -2.1 % | +1.6 % | -0.3 % | -2.3 % |
| 2.0 | 0.8 | -1.0 % | -1.9 % | -2.1 % | -0.6 % |
| 3.0 | 0.5 | -3.2 % | -2.7 % | -0.7 % | -3.4 % |
| 3.0 | 0.8 | -0.8 % | -1.9 % | -2.3 % | -1.4 % |

Same-code legacy controls reproduce the legacy study at AR 3, alpha 0.5
(within 0.4 %) and alpha 0.95 (T* -27.3 vs -26.9 %).  Repair 3 removes most of
the residual N1 and T* deficit left by repairs 1–2, as the operator-level
over-relaxation predicted.  These are single trajectories; the Negishi
campaign supplies four seeds x two branches per case.  Rod DEM at phi = 0.01 also carries Enskog
chi ~ 1.025 and a 1.3–1.7 % collisional shear share that a dilute DSMC does
not have; residuals of that size are not closure error.

## 5. Negishi campaign

`hpc/submit_usf_encounter.sh` (tag once): 440 USF realizations (320 full
model over AR 1.5–3 x alpha 0.50–0.95 x cold/hot x 4 seeds; 80 encounter-only
ablation; 40 same-code legacy controls) on 208 strided single-core workers,
longest rows first, plus 48 HCS runs at the DEM benchmark nodes, and a
dependent analysis job writing `results/usf_encounter_<TAG>/usf_vs_dem.*` and
`results/hcs_encounter_<TAG>/hcs_vs_dem.csv`.

## 6. Negishi campaign result (usf_encounter_v1_20260930)

All 440 USF and 48 HCS realizations completed.  The dependent analysis job
failed only because Negishi's Python environment has no pandas; the analysis
was run locally (`DSMC_0D_v2/scripts/analyze_usf_encounter.py`, and
`DSMC_0D_v2/scripts/analyze_usf_science.py --rod-arm encounter_memory
--encounter-table ... --previous-comparison ...`, which regenerated
`results/usf_dem_dsmc_comparison_20260928`; the legacy version is kept as
`results/usf_dem_dsmc_comparison_20260928_legacy_model`).

Absolute relative difference from DEM over the 40 rod cases (AR 1.5–3,
alpha 0.50–0.95, 4 seeds x cold/hot), legacy -> encounter unit + angular memory:

| observable | median | max |
|---|---|---|
| T* | 8.1 -> 1.1 % | 26.9 -> 7.3 % |
| theta | 9.2 -> 3.0 % | 17.7 -> 7.9 % |
| P*_xy | 8.8 -> 1.5 % | 11.7 -> 2.9 % |
| N1 | 22.4 -> 3.3 % | 30.3 -> 5.4 % |
| kinetic viscosity | 13.2 -> 1.6 % | 22.9 -> 3.3 % |
| collisions per strain (encounters x <k>) | 5.4 -> 1.6 % | 10.2 -> 3.9 % |

HCS at the DEM benchmark nodes: alpha 0.8 AR 2/3 -1.4/-2.1 % (legacy
-2.5/-2.9 %); alpha 0.95 +0.5/+0.8 %; alpha 1 within 0.5 %.

Open: theta is 4–8 % low only at alpha 0.55–0.75, vanishing at the fitted
alpha = 0.5 and 0.8 planes (DEM theta(alpha) has an interior maximum near
0.6), which points to alpha interpolation; AR 1.5 T* is 4.6–7.3 % high at
alpha <= 0.75, present at a fitted node and not yet explained.

## 7. Measured collision cross-section replaces the v1 polynomial

The v1 polynomial `sigma_c(AR)` (from Hong's work, a fit to measured
`sigma(AR)/sigma(1)` ratios times `pi d^2`) was never measured in this project
and lies 2–23 % above the encounter area our CTC data give.  The encounter
model now uses `sigma_c(theta, AR) = A_bar(AR) D(theta, AR)`, with
`A_bar = pi (d^2 + d L + L^2/8)` the exact isotropic mean projected excluded
area and `D` the measured dynamic factor (1.00–1.34), interpolated in
`(log theta, AR)`.  Table: `DSMC_0D_v2/models/encounter_cross_section_v2.json`,
built by `Coll_Models_v2/scripts/build_encounter_table.py`, which counts each
common-random-number attempt stream once (standard errors 0.1–0.3 %).  At
theta = 1, in units of pi d^2: 1.107, 1.222, 1.408, 1.603, 2.323, 3.127, 4.010
for AR 1.1, 1.2, 1.35, 1.5, 2, 2.5, 3.  It agrees with the ratio table used by
the Negishi campaign to 0.40 % at every node, so that campaign stands.  The
legacy contact unit and the alpha = 1 block keep the polynomial for
reproducibility.

## 8. Invariant response retired; alpha refinement; production gate (2026-09-30)

**Invariant response retired.**  The kernel conditions on each pair's own
state, and DSMC collides the actual non-equilibrium particles, so a cell-moment
response can only act through pair variables the kernel does not see.  Tested
on the productions that fix the USF steady state (105 held-out USF replay
windows, common random numbers), both fitted energy responses made the
rotational residual worse (median |r| 0.027 -> 0.038 nonlinear, 0.028 linear;
alpha 0.5 AR 2: 0.001 -> 0.086).  The runtime no longer applies any response,
`invariant_corrections: true` is an error, the flow invariants are computed only
for collision audits, and new artifacts carry no response rows.

**Alpha refinement.**  CTC replays of USF pairs at alpha = 0.65 gave a rotational
residual of +0.07..+0.10 on all four sources, against ~0 at the fitted 0.5 and
0.8 planes: the theta deficit at alpha 0.55–0.75 is interpolation of the energy
law between planes.  The grid gains planes at alpha = 0.55, 0.60, 0.65, 0.70,
0.75, 0.85, 0.90 (252 nodes, 396 in total).  A node's own tables are used on
the node; between nodes (for example alpha = 0.99) the laws are mixed on the
stencil (Wasserstein quantile average for energy, mixture for angle).

**Production gate.**  `DSMC_0D_v2/scripts/production_gate.py` compares the
model with exact CTC on identical USF pairs through the rotational residual,
the dissipation ratio and the stress-production ratios, with batch standard
errors; a replay passes when each metric is within tolerance (0.03, 0.03, 0.04)
or within three standard errors.  On the current artifact it passes every
alpha = 0.5 and 0.8 replay and fails every alpha = 0.65 replay (+0.085..+0.110,
5–7 standard errors).

**Local end-to-end test.**  A slice of the alpha = 0.65 plane (AR 2 and 3,
theta 0.2/1/2; 60,000 encounters and 20 bootstrap resamples per node) went
through the production chain (CTC, fit, deep QA, precompute, pack).  All six
nodes passed QA with the bridge form; the encounter-unit stability gate gives
unique roots theta* = 0.844 (AR 2) and 0.920 (AR 3).  On the same USF pairs the
gate's rotational residual fell from +0.075..+0.094 (fail) to +0.003..+0.022
(pass), mean of four seeds, and the dissipation ratio from 0.961–0.984 to
0.987–1.001 (Section 9 has the seed analysis).  In USF at
alpha = 0.65 (one trajectory, window [96, 160]) theta moved from -7.5 % / -7.2 %
to -1.2 % / -1.3 % of DEM for AR 2 / 3, with T*, P*_xy and N1 within noise.
Per-node cost: fit about 7 h at 200,000 encounters and 200 resamples; precompute
about 30 CPU-minutes.

**Artifact builder.**  The HCS stability gate in the packer used the per-contact
BL mean as the energy destroyed per event; `--event-unit encounter` now uses the
per-encounter CTC mean, as the runtime does.

**Pipeline.**  `TAG=alpha_v1_20261001 bash hpc/submit_alpha_refinement.sh`:
CTC of the new planes and gate replays -> fits of the new nodes (bridge form,
bounded-logit repair on QA failure; the 144 existing fits are reused) -> deep
QA -> precompute -> pack (encounter event unit) -> model tables (sigma, <k>,
angular memory) in the same model folder -> production gate -> USF (320) and
HCS (24) validation -> analysis (pandas-free).  Model config:
`DSMC_0D_v2/config/encounter_unit_model_v2.yaml`.

The v1 model is packaged as `DSMC_0D_v2/config/encounter_unit_model_v1.yaml`.
A rendered derivation, the full algorithm with every closure variable, the
exchange-gate explanation and the nematic-order outlook are in
`reports/usf_event_unit_methodology/usf_event_unit.html`.

## 9. The production measure and the near-elastic yardstick (2026-09-30)

**What the measure is.**  The gate evaluates the four productions that fix a USF
steady state -- D (T* via shear work = D), Lambda_rot
(theta = root of Lambda_rot = 0), Lambda_xy and Lambda_N1 -- with the model and
with exact CTC on the same replayed USF pairs.  It never runs inside a
simulation; it is the acceptance test that selected each model change.  Theta
depends on the energy kernel only through the energy-weighted conditional mean
m(x) = E[(1 - eps)(1 - z') | x]; a shift delta in the mean of z' moves r_rot by
-delta (1 - eps_E)/eps_E, which a Wasserstein distance of the law does not
weight.

**Residual to theta.**  Linear response of the root gives
dtheta/theta = -kappa r_rot, kappa = D/(theta dLambda_rot/dtheta).  To leading
order in the loss, with E[z'|z] = 1/2 + b(z - 1/2), p_exch = 1 - b and flux
energies 2T_tr, 2T_rot: kappa = 2 eps_E (1 + theta)/(theta p_exch), i.e.
4 eps_E/p_exch at theta = 1 (AR 2: 1.45, 0.94, 0.52, 0.13, 0.025 at
alpha = 0.5, 0.65, 0.8, 0.95, 0.99).  Check at alpha = 0.65: previous and
refined models differ by dr_rot = 0.072–0.073 on all four replays (same pairs
and random numbers), predicting a theta rise of 6.9 % (AR 2) and 5.9 % (AR 3);
the USF runs rose by 6.3 and 5.9 points.

**Seeds.**  The earlier gate numbers used 30,000 samples and one seed.  The
model side shares random numbers across replays, so a seed shifts all rows
together by about +-0.01 in r_rot.  Four-seed means (40,000 samples), predicted
theta error -kappa r_rot, against USF theta vs DEM:

| replay | alpha 0.5 | 0.65 previous | 0.65 refined | 0.8 |
|---|---:|---:|---:|---:|
| AR 2, src 0.5 | -0.3 % | -9.3 % | -2.1 % | -2.3 % |
| AR 2, src 0.8 | +0.5 % | -7.0 % | -0.3 % | +0.3 % |
| AR 2 USF vs DEM | -2.9 % | -7.5 % | -1.2 % | -1.3 % |
| AR 3, src 0.5 | -1.9 % | -7.7 % | -1.5 % | -0.7 % |
| AR 3, src 0.8 | +0.1 % | -6.2 % | -0.3 % | -0.5 % |
| AR 3 USF vs DEM | -2.7 % | -7.2 % | -1.3 % | -0.9 % |

At alpha = 0.5 the gate predicts less than USF shows; the replay states there
come from the legacy campaign (theta 0.95–1.11 against a true 1.14–1.17).

**Near the elastic limit.**  D -> 0 as alpha -> 1 (eps_E ~ 0.022 at 0.95, 0.0045
at 0.99), so r_rot and its error grow like 1/eps_E (local alpha = 0.99 replay:
0.97 +- 1.94 with 3,000 pairs) while kappa falls like eps_E.  The gate now
checks the predicted theta error |kappa r_rot| <= 0.02 (or 3 SE) at every alpha
and still reports r_rot, eps_E, p_exch and kappa.  It also replays the
alpha = 0.95 USF states (theta ~ 1.04) at alpha = 0.9, 0.95, 0.975, 0.99
(`make_gate_replay_manifest.py --near-elastic-source`), since the alpha = 0.5
and 0.8 states are dominated near alpha = 1 by relaxation toward equipartition.
A routing error delta in the mean of z' still moves theta by ~4 delta/p_exch at
every alpha.

**Figures.**  `results/usf_dem_dsmc_comparison_20260928/figures/01_state_comparison.png`
and `06_before_after_error.png` now carry the two alpha = 0.65 slice runs as
stars.
