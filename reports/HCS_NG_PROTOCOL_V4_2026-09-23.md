# HCS non-Gaussian protocol v4 - 2026-09-23

## Decision

Protocol v3 is invalid for scientific interpretation. All 200 runs with
`alpha <= 0.8` left the calibrated temperature-ratio domain after a delayed
metastable plateau. The exit time approximately collapsed when expressed as

\[
\chi=(1-\alpha^2)\tau,
\]

with failure near `chi=400..500`. A local `alpha=0.5, AR=2` A/B test failed at
approximately the same time with the angular response both enabled and
disabled and at `N=800` as in the `N=10000` production trajectories. Thus the
domain boundary and angular response are not the cause; they expose a
long-time defect of the iterated energy-partition dynamics. The completed
`alpha=0.9` and `0.95` runs are also provisional because the same scaling can
delay an exit beyond `tau=1500`.

Protocol v4 does not extend the hull or impose a target temperature ratio.
Either action would manufacture a stationary state. It instead fails closed
until the unchanged microscopic model demonstrates a real long-time HCS
attractor.

## Staged gates

1. The 120-task engineering campaign retains the paired scaled/unscaled and
   `dt=0.005`/`0.0025` checks at production particle count. A complete,
   passing v3 engineering summary is reusable only when its artifact hash,
   model variant, correction routing, task count, arms, and all controls match
   exactly; v3 and v4 have the same numerical engineering design. Its
   stationarity diagnostics are reported but are not treated as long-time HCS
   evidence; that authority belongs to the following two stages.
2. A 40-task long-time sentinel first exercises five physical coordinates,
   two initial temperature ratios, two seeds, and both `dt=0.005` and
   `0.0025`. This rejects a defective model before paying for a full-domain
   allocation.
3. Only after the sentinel passes, a 148-task stability campaign covers all
   37 production coordinates. Each coordinate starts below and above its
   predicted root, with two independent seeds and `N=2000`: the low start is
   `Ttr/Trot=0.025` through `AR=1.35` and `0.225` at larger aspect ratios,
   just inside the calibrated lower boundary of `0.2`; the high start is
   `1.5` throughout.
4. Every inelastic stability task runs to `chi=600`. Elastic tasks run to
   `tau=1500`.
5. The gate requires complete output, a five-block/change-point stationarity
   test, agreement of the two late-time attractors, the full dissipation
   horizon, a positive calibrated-domain margin, runtime/NTC quality, and
   long-horizon time-step consistency.
6. Sweep and map campaigns require the passing stability summary from the
   exact same artifact bytes. The 1,200-task tail campaign additionally
   requires a complete, passing 370-task sweep, so the largest allocation
   cannot run after a production-level failure.

An exception now writes `*.failed.json` and flushes the partial moments and
histograms with status `aborted_not_stationary`. Partial data are retained for
diagnosis but are never admitted into an HCS estimate.

## Non-Gaussian analysis

The scientific analyzer now:

- writes both temperature-ratio conventions and gates `log(Ttr/Trot)`;
- performs stationarity checks on ensemble-mean late-time curves rather than
  accepting a fixed `tau=500` burn-in;
- releases figures only after the whole scientific campaign passes;
- compares observables both along `alpha` at fixed aspect ratio and along
  aspect ratio at fixed `alpha`;
- plots the actual vector marginals `phi_c(c)`, `phi_w(w)`, and
  `phi_cw(c^2 w^2)` with the Maxwellian references for `dt=3, dr=2`;
- separately plots the full fourth-order Sonine ratios specialized to
  `dt=3, dr=2`;
- uses exponential fits for the translational tail and algebraic fits for the
  rotational and product tails, as in Megias and Santos (2023);
- requires tail-count, occupied-bin, threshold-stability, jackknife,
  fit-quality, and exact-elastic negative-control gates before marking an
  asymptotic claim ready.

The exact `alpha=1` Hong-Morris block remains a separate singular control and
is not used as a point in an inelastic closure fit.

## Required submission order

The engineering, stability-sentinel, stability, production-sweep, and tail
stages must be submitted in that order. Inspect each generated `summary.json`;
do not bypass a failed gate.
The precise shell commands are included in the handoff response associated
with this protocol and in `hpc/submit_hcs_ng_campaign.sh --help`-equivalent
usage in the script header.
