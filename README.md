# DSMC_V2 conservative microscopic closures

`DSMC_V2` is one repository containing the collision-trajectory generator, the
standalone estimators, and the 0D DSMC consumer for smooth spherocylinders.
Version 2.1 deliberately preserves the proven version-1 collision frequency
and scalar energy model. It changes only dissipation routing and angular
scattering.

No HCS, USF, Fourier, DEM, or LAMMPS ensemble result is used to estimate the
new coefficients. Those simulations are held-out validation cases.

## Architecture

```text
HS_CTC_v2
  flux-weighted incoming pairs
  attempts (hits + misses) and accepted outcomes
             |
             v
Coll_Models_v2
  BL-compatible 16-moment routing estimator
  direction-only VSS rank-2 estimator
  rotational direction donor library
             |
             v
models/microscopic_closure_v2/
  routing16_v2.json
  vss_rank2_v2.json
  rotational_direction_v2.npz
             |
             v
DSMC_0D_v2
  unchanged v1 NTC + sigma_c + GMM + BL total loss
  new F_tr split + new outgoing direction
```

The repository packages have separate jobs:

- `HS_CTC_v2` measures isolated two-particle collisions.
- `Coll_Models_v2` turns those measurements into DSMC-ready microscopic
  closures and uncertainties.
- `DSMC_0D_v2` runs HCS or homogeneous USF using the old scalar collision
  model plus the two optional v2.1 closures.
- `contracts` is the single definition of the binary records and 16 features.
- `hpc` contains pilot, production, continuation, estimation, and aggregation
  Slurm workflows.

## What is preserved

The production DSMC path retains the following version-1 behavior:

- Candidate count

  ```text
  N_cand = N(N-1) sigma_c g_max dt / V.
  ```

- Uniform collision-normal proposals and acceptance

  ```text
  P_acc = |g dot e| / g_max.
  ```

- The fixed cross-section, with no fitted multiplier,

  ```text
  sigma_c(AR,D) = pi D^2 (0.32 AR^2 + 0.694 AR - 0.0213).
  ```

- The six committed conditional-GMM files for `AR = 1.1, 1.25, 1.5, 2.0,
  2.5, 3.0`.
- `Zr`, GMM relaxation selection, `gamma_max`, `P1hit`, the `Beta(1.21,3.67)`
  draw, negative-reservoir transfer, the sphere bypass, HCS time output, USF
  shear drift, and USF pressure accumulation.

The exact v1 NTC source and conditional-GMM implementation are imported from
commit `cd6340b`. `SOURCE_VERSIONS.md` records their provenance.

There is no pair-clock model, state-dependent cross-section, total-loss moment
correction, cKDTree replacement for the GMM, or constrained post-energy
reconstruction in the v2.1 production path.

## CTC measure and accepted sample count

CTC draws the normalized incoming-flux proposal

```text
q0(z1,z2,b) = phi0(z1) phi0(z2) g / (G0 A0),  b uniform in A0.
```

The collision-weighted speed sampler uses `g^2/4 ~ Gamma(2,1)`. The pair
center-of-mass velocity is Maxwellian, and a Haar-uniform global rotation makes
the complete laboratory state isotropic.

`NSAMPLES` means accepted collisions. If a node requests 100,000 samples,
`outcomes_v2.bin` contains exactly 100,000 records. CTC may attempt more than
100,000 trajectories; every intervening miss is retained in `attempts_v2.bin`.
This is necessary because the routing numerator is a hit production while its
preserved-BL denominator is an all-proposal production.

The CTC cross-section estimate

```text
sigma_CTC = A0 N_hit / N_try
```

is reported only as QA against the frozen polynomial. It never changes the
DSMC clock.

## DSMC-compatible routing estimator

For proposal `j`, the unchanged BL mean fractional loss is

```text
mean_gamma = gamma_max P1hit a_gamma/(a_gamma+b_gamma)
delta_BL,j = E_i,j mean_gamma.
```

The reference routing fraction is

```text
F0 = A0 sum_try(H delta_tr,CTC)
     / [sigma_c sum_try(delta_BL)].
```

The estimator forms the agreed 16 pair scores `K_a` from every incoming state:

```text
L_tr,a = sum(H delta_tr K_a) / sum(H delta_tr)
L_BL,a = sum(delta_BL K_a) / sum(delta_BL)
beta_a = L_tr,a - L_BL,a.
```

The runtime uses the derivative-equivalent bounded form

```text
logit(F_tr) = logit(F0) + sum_a eta_a X_a,
eta_a = beta_a/(1-F0).
```

This guarantees `0 < F_tr < 1` without clipping. It changes only how the
already drawn total loss is divided between translation and rotation.

The artifact also contains the pure-CTC audit

```text
F_C = sum(H delta_tr) / sum(H delta),
beta_C,a = <K_a>_(H delta_tr) - <K_a>_(H delta).
```

and a total-loss compatibility ratio. Artifact release stops if the frozen BL
loss production differs from CTC by more than 10% after uncertainty is
included. This is intentional: routing
must not hide an error in a preserved total-loss kernel.

The 16 variables and normalizations are documented in
`contracts/FEATURE_BASIS.md`. Tensor/vector squares and cross-products use
finite-population U-statistics in DSMC, removing the false `O(1/N)` anisotropy
of squared noisy cell means.

## VSS rank-2 scattering

Accepted CTC directions provide

```text
B2(alpha,AR) = mean[1 - P2(ghat dot ghat')].
```

The forward VSS exponent is obtained directly from

```text
B2 = 6 alpha_eff / [(alpha_eff+1)(alpha_eff+2)].
```

No energy, temperature, routing fraction, dissipation weight, GMM quantity, or
`p_eta` enters this fit. `P1`, `P3`, and `P4` are exported as diagnostics. A
`B2` outside the forward-VSS family fails QA rather than being clipped.

The VSS draw changes only `ghat'`. Its magnitude remains the value produced by
the unchanged GMM/BL energy kernel. Runs at other theta values test the assumed
temperature independence; they are not extra VSS inputs.

## Rotational vector state

Each DSMC particle now retains velocity, an axis, tangent angular velocity, and
the original scalar rotational energy. Axes advance using

```text
du/dt = omega cross u.
```

The scalar GMM energy is authoritative. After it sets the two post-collision
rotational energies, the direction-only CTC library supplies paired unit spin
directions using 64 inverse-distance-weighted neighbours in an equivariant
pair frame. The directions are projected onto the two tangent planes and then
scaled to satisfy

```text
E_r,p = (I/2) |omega_p|^2.
```

Axes do not jump during an instantaneous collision. Separate random streams
are used for axes, spin donors, and VSS, so passive vector bookkeeping does not
alter the legacy NTC/GMM random stream.

Routing and VSS statistics use every event. The deployed direction library is
a deterministic node-stratified subset capped at 500,000 donors so its cKDTree
has bounded memory; donor coverage remains an explicit QA diagnostic.

## Build and local verification

From the repository root:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python3 -m pip install -e contracts -e Coll_Models_v2 -e DSMC_0D_v2
make -C HS_CTC_v2/build clean all
make test
```

A small CTC run is:

```bash
tmp_run=$(mktemp -d /tmp/ctc_v21.XXXXXX)
cd HS_CTC_v2
OMP_NUM_THREADS=1 ./build/SphCyl 0.8 1.0 1.0 1.0 "$tmp_run" 123 32 v2
PYTHONPATH=../contracts/python python3 scripts/finalize_run.py "$tmp_run"
cd ..
```

For DSMC regression, set both closure switches to legacy. For ablations run,
in order: all legacy, routing only, VSS only, then both.

## Production sweep

Each pilot or production shard contains 870 CTC nodes:

- routing: 12 alpha values (`0.50:0.05:0.95, 0.975, 0.99`) by 12 theta values
  (`0.1:0.1:1.2`) by 6 aspect ratios;
- VSS elastic references: `alpha=1, theta=1` at the same six aspect ratios.

The sphere is a separate analytic bypass and validation case. `AR=4` is not in
this release because no unchanged v1 GMM exists for it.

Sampling proceeds as follows:

1. Pilot: 20,000 accepted outcomes per node.
2. Production: a separate 80,000-outcome shard, giving 100,000 accepted
   outcomes after aggregation.
3. Continuation: 100,000-outcome shards only for failing nodes, up to one
   million accepted outcomes.
4. Uncertainty: 128 deterministic attempt-stream blocks and 2,000 bootstrap
   replicates, with score-tail and leave-one-shard-out diagnostics.

Seeds depend on `(theta, AR, shard)`, not alpha, so an alpha line uses common
random numbers.

## Slurm workflow

On the cluster login node:

```bash
git clone YOUR_REMOTE DSMC_V2
cd DSMC_V2
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python3 -m pip install -e contracts -e Coll_Models_v2 -e DSMC_0D_v2
make -C HS_CTC_v2/build clean all
mkdir -p logs results manifests coefficients models
```

Generate and submit the accepted-hit pilot and production shards:

```bash
python3 hpc/make_manifest.py --stage pilot --output manifests/pilot.csv
bash hpc/submit_manifest.sh manifests/pilot.csv

python3 hpc/make_manifest.py --stage production --output manifests/production.csv
bash hpc/submit_manifest.sh manifests/production.csv
```

Only directories containing `_SUCCESS` are estimated. Build the grouped
estimation manifest and submit at most 50 simultaneous estimators:

```bash
PYTHONPATH=contracts/python:Coll_Models_v2/src \
python3 hpc/make_estimation_manifest.py \
  --runs-root results/ctc --output manifests/estimate.csv

N=$(( $(wc -l < manifests/estimate.csv) - 1 ))
sbatch --array="0-$((N-1))%50" hpc/estimate_array.slurm \
  manifests/estimate.csv coefficients/node_estimates
```

Create the QA table and continuation manifest:

```bash
PYTHONPATH=contracts/python:Coll_Models_v2/src \
python3 Coll_Models_v2/scripts/estimate_grid.py \
  --runs-root results/ctc --output coefficients --bootstrap 2000 \
  --gamma-max-table Coll_Models_v2/models/legacy_bl/gamma_max_table.json \
  --one-hit-table Coll_Models_v2/models/legacy_bl/one_hit_table.json

python3 hpc/make_manifest.py --stage continuation --shard 2 \
  --qa-summary coefficients/qa_summary.csv \
  --output manifests/continuation_02.csv
bash hpc/submit_manifest.sh manifests/continuation_02.csv
```

Repeat with increasing shard numbers until all nodes pass or reach the cap.
Then build the three-file microscopic closure bundle:

```bash
sbatch hpc/aggregate.slurm results/ctc models/microscopic_closure_v2
```

## Running DSMC

Select either closure independently in `DSMC_0D_v2/config/default.yaml`:

```yaml
microscopic_closure:
  routing: ctc_moment16       # or legacy_rank0
  angular: ctc_vss_rank2      # or legacy
  routing_artifact: models/microscopic_closure_v2/routing16_v2.json
  vss_artifact: models/microscopic_closure_v2/vss_rank2_v2.json
  rotational_direction_artifact: models/microscopic_closure_v2/rotational_direction_v2.npz
```

Run HCS or set `flow.mode: usf` with a nonzero `shear_rate`:

```bash
PYTHONPATH=contracts/python:Coll_Models_v2/src:DSMC_0D_v2/src \
python3 DSMC_0D_v2/scripts/run_simulation.py \
  --config DSMC_0D_v2/config/default.yaml
```

The temperature output remains `t, cpp, Ttr, Trot, Ttotal`. USF additionally
writes the v1 kinetic and collisional pressure-tensor columns.

## Acceptance gates and limitations

Release is blocked by corrupt/incomplete binary data, polynomial cross-section
disagreement above the specified uncertainty criterion, preserved-BL
total-loss mismatch, unresolved routing/VSS uncertainty, unstable score tails,
an unrepresentable VSS target, failed theta-independence diagnostics, or failed
one-collision routing/stress weak forms.

The routing model is a first-order 16-moment closure. Finite-amplitude held-out
tests remain necessary. Matching `P2` alone does not prove that every angular
moment or full nonlinear stress production is correct. The direction donor
library is also an empirical conditional approximation. These limitations are
checked with one-collision tests first and HCS/USF/Fourier/DEM ensembles only
afterward as validation—not fitting data.

All packages and schemas are version `2.1.0`. The release tag is created only
after the full acceptance suite passes. Commits and tags use only the
repository owner's configured Git identity and contain no AI co-author or
contributor trailers.
