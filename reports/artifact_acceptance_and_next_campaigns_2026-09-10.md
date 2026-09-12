# Artifact acceptance and next campaigns — 2026-09-10

## Verdict

The copied schema-2.3 artifact is numerically acceptable for the baseline HCS
campaign with invariant corrections disabled.  Its smaller size does not come
from lossy compression: the arrays occupy 61,319,972 bytes before ZIP and the
archive occupies 56,999,636 bytes, only a 1.076 compression ratio.  The large
reduction comes from replacing 2,049 uniform `a` rows at every node with the
33--181 rows that each conditional law actually needs.

This establishes faithful numerical representation of the fitted collision
law.  It does not, by itself, prove that the fitted law reproduces a dynamical
DSMC trajectory; that is the purpose of the HCS gate below.

## Independent checks on the copied files

- All 144 Slurm precompute tasks produced a node payload and all 144 stderr
  logs are empty.
- The final ZIP/NPZ CRC test passes.
- The copied-file SHA-256 digests are
  `b043fa89cf2ce2486674898d302c28acbc243f04d2ac27622889fb110bb5b706`
  for `closure_v2.npz` and
  `3ec4dba43e27395bcff5b079b1015149d5e462fd2caa3bdcc006e91d2740203d`
  for `manifest.json`.  These identify the exact accepted pair for later
  provenance checks.
- Every packed node table and collision-measure curve is byte-for-byte equal
  to its physical precompute payload.
- Every payload digest matches the current accepted node-estimate JSON.
- The embedded loss and NTC-clock hashes match the current frozen BL tables,
  `ntc.py`, and `particle.py`.
- All energy quantiles lie in `[0,1]`; all `a` grids and probability rows are
  monotone; all collision enhancements are finite and positive.
- The enhancement range is 0.7909--1.2797.  It changes pair selection but is
  normalized per node so the pre-existing NTC clock remains authoritative.
- The builder's largest adaptive midpoint error is
  `1.999387e-4`.  An independent quarter/three-quarter-point audit over every
  final interval found `6.597053e-5` as the largest quantile error.
- The maximum energy-quantile moment error is `9.764036e-4`; the maximum
  angular moment error is `1.978864e-5`.  Both are absolute errors and pass the
  artifact's `1e-3` release bound.
- All 28 analytic HCS families have one root with negative drift derivative.
  The maximum elastic-root error is 0.00216 (0.216%).

For the six blocking HCS comparisons, analytic roots differ from the DEM/exact
targets by +0.030%, +0.157%, +0.551%, +0.698%, -1.920%, and -2.595%.  Those are
comfortably inside the 10% experimental/dynamical acceptance band, but the
finite-particle DSMC trajectories must still confirm attraction and bounded
noise.

## HCS gate

The HCS campaign now uses three statistically independent replicates from each
of two initial partitions for every one of six gate cases: 36 independent
one-core tasks.  Each trajectory uses 2,000 particles and 20 collisions per
particle.  The plot job runs automatically after all trajectories finish and
shows the replicate mean with a one-standard-deviation band.

The gate distinguishes the correct physics:

- for `alpha < 1`, total granular energy must remain positive and decrease;
- for `alpha = 1`, total energy must remain within 2% of its initial value;
- the two initial partitions must converge within 10%;
- replicate coefficient of variation, late drift, and DEM target error must
  each be at most 10%;
- negative-energy repairs, energy-table clamps, and material out-of-domain
  use are forbidden.

The newly reported `energy_axis_clamps` makes the adaptive-table support
directly observable.  Zero clamps is required; this is the end-to-end check
that no runtime collision reaches an omitted part of the old rectangular
table.

## Excitation campaign

Excitation is released only if the baseline HCS physics gate passes.  The first
campaign is a 96-task isotropic pilot at `(alpha, AR) = (0.8,2), (0.8,3),
(0.95,2), (0.95,3), (1,2), (1,3)`, all at `theta=1`.  At each node it excites
`a2_tr`, `a2_rot`, `a11`, and `A_cu` in four directions
`eta = -0.5,-0.25,0.25,0.5`.

These are exact importance-sampled virtual ensembles under molecular chaos:

\[
  \mathbb E_\eta[F]
  = \frac{\mathbb E_0[r_\eta(x_1)r_\eta(x_2)F]}
         {\mathbb E_0[r_\eta(x_1)r_\eta(x_2)]}.
\]

No collision trajectory or contact law is changed, so no fresh CTC is needed
for the pilot.  Every task nevertheless measures ESS and maximum normalized
weight and fails closed below ESS/N = 0.5 or above a 1% maximum share.

The amplitudes were reduced from 0.6 to 0.5 after a real 200k-event audit:
some `eta=0.6` pair ensembles retained only ESS/N = 0.493.  At `|eta| <= 0.5`,
the full 72-row, 14-feature pilot at `(0.95,1,2)` has rank 14, scaled condition
number 2.41, minimum ESS/N 0.604, and maximum weight share `7.78e-5`.

The pilot regresses and reports responses of `p_exch`, all available energy
natural parameters, and angular parameters.  It does not assume that only
`lambda1` responds and it never modifies the deployable artifact.  Only after
the response report identifies the statistically material parameters should a
production correction surface and its enlarged adaptive `a` support be built.

## Negishi sequence

```bash
git pull --ff-only origin closure-v2-repairs
bash hpc/submit_hcs_validation.sh
```

After `hcs_plot_job` completes:

```bash
jq '{physics_gate_pass,production_gate_pass,cases}' \
  results/hcs_validation/summary.json
ls -lh results/hcs_validation/hcs_theta_attraction.png
```

Only if `physics_gate_pass` is true:

```bash
bash hpc/submit_excitation_campaign.sh hcs-pilot
```

The full anisotropic rank pilot is prepared but should not be launched before
the HCS pilot is interpreted:

```bash
bash hpc/submit_excitation_campaign.sh full-pilot
```

Neither excitation command launches fresh CTC or rebuilds the production
artifact.  That separation prevents an untested first-order assumption from
entering USF.
