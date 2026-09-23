# HCS non-Gaussian protocol v6 - 2026-09-23

## Decision

Protocol v5 permanently removed the false late-time temperature-ratio runaway:
all 40 long-time sentinel tasks completed, reached their requested horizons,
retained positive physical-theta hull margins, and kept the bulk-to-thermal
temperature ratio below `5.2e-32`. All five low/high initial-temperature pairs
reached the same late attractor. The v5 sentinel nevertheless failed three
independent quality controls and must not authorize the 148-task allocation.

The failures and v6 repairs are:

1. **Low-theta NTC initialization.** The majorant used initial `Ttr`, although
   rotational energy can heat translation while total rescaled energy remains
   fixed. V6 initializes it from the energy-conserving bound
   `Ttr,max = Ttr,0 + 2 Trot,0 / 3`. A low-theta smoke test then retained
   `vrmax/vrmax_initial=1` with zero majorant violations.
2. **Correction-feature extrapolation.** Rare `a2_tr` excursions crossed the
   measured response-feature box. V6 never extrapolates a learned response:
   the affected state update uses the calibrated base collision law. It
   reports fallback over the whole trajectory and separately over the retained
   statistical window. The window fails at a fallback fraction of one percent
   or more; this threshold limits fallback to a negligible safe mixture rather
   than allowing an unsupported correction.
3. **Noise-blind stationarity.** The old five-block absolute-change rule
   rejected 19/20 groups, including exact elastic equilibrium. V6 tests the
   signed early-to-late drift across independent realizations. A drift must be
   consistent with zero or the declared practical floor, and its three-SE
   resolution must independently remain below `0.06` for `log(theta)` and
   `0.04` for fourth-order observables. Thus large uncertainty cannot create a
   pass.
4. **Long-time step convergence.** At `alpha=0.5, AR=1.35`, all four paired
   `A_cw` differences between `dt=0.005` and `0.0025` had the same sign; the
   mean absolute difference `0.002828` exceeded its three-SE tolerance
   `0.002434`. V6 promotes `dt=0.0025` to production and compares it with
   `dt=0.00125`.
5. **Insufficient convergence precision.** Two sentinel seeds left the paired
   `a02` comparison under-resolved (`3 SE=0.0116` versus a `0.01` limit). V6
   uses four seeds, making an 80-task sentinel.

## Staged allocation

1. Run the new 120-task engineering campaign. Earlier engineering summaries
   are incompatible because the time-step arms changed.
2. Only a passing engineering summary may launch the 80-task long-time
   sentinel: five coordinates, two initial temperature ratios, four seeds, and
   both `dt=0.0025` and `0.00125`.
3. Only a complete passing sentinel may launch the 148-task two-sided
   full-domain stability campaign at `dt=0.0025`.
4. Only a complete passing stability result may launch the 370-task production
   sweep at `dt=0.0025`.
5. Tail allocation remains gated on the complete production sweep.

Protocol v5 outputs remain diagnostic evidence for the repairs but are never
mixed into a v6 scientific estimate.
