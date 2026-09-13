# Full-domain closure completion plan — 2026-09-13

## Diagnosis of the completed candidate

The candidate artifact was successfully built and the independent fresh-CTC
holdout passed (36/36 observations, zero failures).  Its compact HCS dynamics
are bounded, cool/conserve energy correctly, and remain within the accepted
10% target and attraction tolerances.  Sixteen of 36 realizations sampled at
least one state outside the fitted `a2_tr`/`a2_rot` hull, and 13 exceeded the
0.1% out-of-domain gate; that support failure is why the HCS physics verdict
was false.  No energy-axis clamp, monotonic repair, or negative-energy repair
occurred.

The existing correction surface contains 36 physical nodes.  Replaying all
2,592 completed excitation observations with the corrected distribution-law
emulator gives:

- parameter response pass: true;
- distribution response pass: true;
- offline validation pass: true;
- 20/36 nodes independently support expansion from |eta|=0.25 to 0.5;
- 16/36 nodes remain restricted to 0.25.

The six original HCS gate coordinates at alpha={0.8,0.95,1.0}, AR={2,3}, and
theta={1,2} support the expanded 0.5 hull.  Expansion is local; failed nodes
are never relaxed globally.

## Corrections required before deployment

1. Complete the correction surface at all 144 baseline artifact coordinates.
   The 36 completed coordinates are reused, leaving 108 new physical nodes.
2. Add independent |eta|=0.35 response points.  These provide an observed
   intermediate support boundary where 0.5 is too nonlinear; coefficients
   remain fitted only at |eta|=0.25.
3. Correct the bounded-logit memory mechanics: at those nodes lambda3 changes
   the first bounded-logit memory coefficient, rather than being rejected as
   a non-Sinkhorn correction.
4. Build one coefficient-surface cache.  Artifact array tasks read 144
   baselines plus this cache instead of each rereading and refitting about
   15,000 excitation JSON files.
5. Bound adaptive energy tables over each node's adjacent interpolation star.
   This preserves all reachable corrections without applying one extreme
   low-temperature correction interval to every table.

## Performance repair and accuracy guard

The old approximately 70% `closure_overhead_fraction` included both cell-state
reduction and the variational collision law itself.  Diagnostics now separate:

- `closure_state_seconds`: feature reduction and state interpolation;
- `closure_collision_seconds`: the deployed per-collision law;
- `closure_total_fraction`: both;
- `closure_overhead_fraction`: state bookkeeping only.

The fourteen features were algebraically reduced from per-particle 3x3 tensor
arrays to identical aggregate contractions.  Random-state equivalence tests
agree to at most 1.1e-16.  On a local 400-particle, 5-collision-per-particle
HCS benchmark, a 0.05-cpp refresh reduced runtime from 4.76 s to 1.49 s and
state overhead from 69.6% to 10.4%.  This timing result does not authorize the
cadence physically.

Negishi therefore runs 168 every-step references and 168 seed/design-matched
0.05-cpp cases: every one of the 28 artifact alpha/AR nodes, two initial
partitions, and three replicates.  The optimized arm is also the full-domain
HCS gate, so it is not run a third time.  Acceptance is based on
three-replicate mean theta and cooling curves, late theta, at least 2x runtime
speedup, at least 10x fewer state updates, and less than 15% state overhead.
USF and non-Gaussian jobs depend on that paired check and cannot start if it
fails.

## One-shot Negishi dependency chain

The submission script creates 15,552 total excitation observations over all
144 nodes by pooling 2,592 completed observations, 1,296 intermediate-support
observations at completed coordinates, and 11,664 observations at the 108
missing coordinates.  Each expensive fit is one Slurm task; jobs are split
into MaxArraySize-safe chunks so no worker serially performs 8–12 fits and
hits the 24-hour wall limit.

After correction QA, the chain builds the artifact with 128 simultaneous
two-core tasks, then runs:

- full-domain every-step and optimized HCS cadence arms (168 + 168), with the
  optimized arm also serving as the corrected HCS gate;
- fresh-CTC artifact rescore (no new CTC trajectory);
- paired USF pilot (72, spanning AR=1.1,2,3), then the 280-case full USF grid
  over all seven nonspherical aspect-ratio nodes only if authorized;
- 112-case non-Gaussian all-domain timing/stationarity pilot after full HCS.

The artifact's nonspherical interpolation domain is the measured grid
1.1 <= AR <= 3 and 0.5 <= alpha <= 1.  AR=1 is the separate exact sphere
kernel/control, not extrapolation of the nonspherical closure.
