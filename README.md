# DSMC_V2

DSMC_V2 is a clean, versioned CTC-to-DSMC monorepo for a dilute granular gas of
smooth spherocylinders. Collision operators are identified directly from
binary collision trajectories. HCS, USF, Fourier, DEM, and LAMMPS ensemble
runs are validation cases only; they are not coefficient-fitting inputs.

## Architecture

```text
HS_CTC_v2
  incoming-flux attempts + accepted full outcomes
             │
             ▼
Coll_Models_v2
  clock ─ joint energy ─ VSS rank-2 ─ 16-moment surfaces
             │
             ▼
models/collision_operator_v2
             │
             ▼
DSMC_0D_v2
  pair_resolved (default)  OR  moment16 (reduced)
```

The repository is intentionally not a copy of the three old project trees.
`SOURCE_VERSIONS.md` records the committed v1 source revisions from which the
minimal mechanics and entry points were imported. Old fitted artifacts,
notebooks, result folders, and calibration branches are absent.

## The four operator layers

### 1. Collision measure and clock

CTC proposals sample

```text
dmu_try proportional to phi(z1) phi(z2) g dz1 dz2 db.
```

Both hits and misses are retained. Consequently

```text
sigma_eff = A0 * N_hit / N_try
```

has the same incoming-flux normalization used by the DSMC NTC clock. The
pair-resolved model fits `Pr(hit | incoming pair state)`. The reduced model
uses separately estimated 16-score derivatives. These corrections are never
multiplied together.

The NTC implementation samples distinct unordered pairs uniformly. It uses
`A0` as the exact cross-section bound and a finite-population speed bound based
on the cell's total internal energy. A majorant violation raises an error; it
is never hidden by probability clipping.

### 2. Energy distribution and routing

The old product of an event maximum, a Beta draw, a one-hit probability, and an
independent GMM is not used. CTC outcomes are stored as a joint conditional
library of retained energy, translational/rotational composition, per-particle
rotational split, contact count, and effective geometry. A `cKDTree` query
resamples this joint state without negative-reservoir clipping.

The reduced closure estimates independent response vectors for the total loss
and translational routing means. Bounded means use logit-space responses;
`Gamma(alpha=1)=0` is exact. For a sphere, `Ftr=1` and orientation terms are
irrelevant.

### 3. Scattering

Scattering uses only the direction-only VSS rank-2 match

```text
B2 = <1 - P2(ghat_pre dot ghat_post)>
   = 6 alpha_eff / ((alpha_eff + 1)(alpha_eff + 2)).
```

There is no `p_eta`. The VSS artifact is indexed only by `(alpha, AR)` and its
schema explicitly forbids temperature, energy, dissipation, and `f_tr` inputs.
The larger forward root is used. A target outside the VSS family fails QA
instead of being clipped.

This is an angular-model separation, not a claim that the complete stress
production is automatically independent of energy. The composed kernel must
pass the held-out one-collision stress weak-form test. `P1`, `P3`, and `P4` are
diagnostics; `P2` is the fitted stress-transport moment.

### 4. Full post-state reconstruction

VSS supplies the outgoing relative direction and the energy library supplies
the retained energy. DSMC then solves for a full post state that preserves pair
center-of-mass velocity, total angular momentum, tangent angular velocities,
and the sampled total energy. Infeasible draws are resampled and counted. No
energy is clipped. Production is rejected if the infeasible fraction is not
statistically negligible.

## Data contracts

`contracts/event_schema_v2.json` freezes the binary layout:

- `attempts_v2.bin`: 200 bytes/record, including typed IDs and hit/miss flag.
- `outcomes_v2.bin`: 552 bytes/accepted collision, including complete pre/post
  vectors, energy identities, geometry, and incoming/outgoing directions.
- `attempt_blocks_v2.csv`: 128 deterministic event-stream blocks with clock,
  loss, routing, angular, and all 16 score sums.
- `metadata_v2.json`, `qa_v2.json`, `_RAW_SUCCESS`, and `_SUCCESS` distinguish
  raw completion from a validated run.

`contracts/FEATURE_BASIS.md` defines the common feature order. Estimator and
runtime use the same Python implementation. Tensor/vector squares and cross
products are U-statistics rather than squares of noisy sample means.

## Local setup and verification

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python3 -m pip install -e contracts -e Coll_Models_v2 -e DSMC_0D_v2
make clean all
make smoke
```

The smoke target builds Fortran, generates a small sphere dataset, validates
both binary streams, and estimates one node. A tiny sample may correctly report
that its noisy `B2` is not VSS-representable; artifact production remains
blocked until the production confidence interval resolves the target.

## Production grid and HPC workflow

The inelastic grid contains 1,200 nodes per shard:

- `alpha = 0.50, 0.55, ..., 0.95, 0.975, 0.99`
- `theta = 0.1, 0.2, ..., 2.0`
- `AR = 1.0, 1.5, 2.0, 3.0, 4.0`

Each shard adds five elastic `alpha=1, theta=1` scattering/clock reference
nodes. Seeds depend on `(theta, AR, shard)` but not alpha, preserving common
random numbers along alpha lines.

On the HPC login node:

```bash
git clone YOUR_DSMC_V2_REMOTE DSMC_V2
cd DSMC_V2
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python3 -m pip install -e contracts -e Coll_Models_v2 -e DSMC_0D_v2
make build
mkdir -p logs results manifests coefficients models
```

Pilot and production:

```bash
python3 hpc/make_manifest.py --stage pilot --output manifests/pilot.csv
bash hpc/submit_manifest.sh manifests/pilot.csv

python3 hpc/make_manifest.py --stage production --output manifests/production.csv
bash hpc/submit_manifest.sh manifests/production.csv
```

After jobs finish, every usable run must have `_SUCCESS`, not merely
`_RAW_SUCCESS`. Generate the estimation array:

```bash
PYTHONPATH=contracts/python:Coll_Models_v2/src \
python3 hpc/make_estimation_manifest.py \
  --runs-root results/ctc --output manifests/estimate.csv

N=$(( $(wc -l < manifests/estimate.csv) - 1 ))
sbatch --array="0-$((N-1))%50" hpc/estimate_array.slurm manifests/estimate.csv
```

Create `qa_summary.csv` and request only failing continuations:

```bash
PYTHONPATH=contracts/python:Coll_Models_v2/src \
python3 Coll_Models_v2/scripts/estimate_grid.py \
  --runs-root results/ctc --output coefficients --bootstrap 2000

python3 hpc/make_manifest.py --stage continuation --shard 2 \
  --qa-summary coefficients/qa_summary.csv \
  --output manifests/continuation_02.csv
bash hpc/submit_manifest.sh manifests/continuation_02.csv
```

Repeat continuation shards, increasing `--shard`, until nodes pass or reach
one million accepted outcomes. Then build the consumer artifact:

```bash
sbatch hpc/aggregate.slurm results/ctc models/collision_operator_v2
```

Run DSMC after setting the artifact path and desired mode in
`DSMC_0D_v2/config/default.yaml`:

```bash
python3 DSMC_0D_v2/scripts/run_simulation.py \
  --config DSMC_0D_v2/config/default.yaml
```

## Acceptance gates and limitations

Production is blocked when any of the following occurs:

- binary corruption, missing hits/outcomes, energy identity failure, or bad
  unit/tangent vectors;
- unresolved clock or core coefficient uncertainty;
- a direction-only `B2` outside the forward VSS family;
- measurable theta dependence in held-out scattering runs;
- non-negligible reconstruction infeasibility;
- failure of CTC-versus-DSMC one-collision clock, loss, routing, stress, or
  16-moment weak forms.

The present scope is dilute molecular-chaos DSMC for axisymmetric smooth
spherocylinders. Fourier data require a spatial/multicell DSMC driver and are a
held-out operator validation target, not a feature of this 0D driver.

## Versioning

All packages and schemas are `2.0.0`. The repository release tag is `v2.0.0`.
Commits use the repository owner's configured Git identity and contain no AI or
Codex co-author/contributor trailers.

