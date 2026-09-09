# Post-fit HCS verdict — 2026-09-09

## Verdict

The closure-estimation stage is complete. All 144 canonical node estimates are
current and pass deep QA, including the nine repaired low-theta fits. Those nine
should remain in the artifact; there is no scientific reason to discard them.

The end-to-end DSMC verdict is still pending for one operational reason: Slurm
never started the artifact builder. `aggregate.slurm` inferred the repository
from `$0`, but Slurm executes a copied script under `/var/spool/slurm`. The job
therefore looked for `/var/spool/slurm/hpc/python.sh` and stopped on line 19.
This has been fixed by resolving the checkout from `SLURM_SUBMIT_DIR`.

The local `models/microscopic_closure_v2` directory is not the new artifact. It
contains a 24-node artifact dated September 7. The new artifact must contain the
full 144-node grid and must be built before its HCS trajectories are judged.

## What “artifact” means

The raw CTC shards are collision observations. The per-node JSON files are
fitted conditional collision laws. The **artifact** is the compiled runtime
package made from all accepted node fits:

- `closure_v2.npz`: arrays, interpolation tables, conditional quantiles, and
  kernel coefficients consumed by DSMC;
- `manifest.json`: provenance, grid coverage, schema, stability gates, and
  numerical-error summaries.

It is analogous to a trained-model checkpoint plus its model card. It does not
replace the collision data; it packages the fitted laws so the DSMC can query
one coherent kernel at arbitrary supported `(alpha, theta, AR)`.

## Correct HCS acceptance test

An inelastic homogeneous cooling state has no nonzero steady total granular
temperature. Energy continues to decay. A dimensionless energy partition can,
however, approach an asymptotic scaling state:

\[
  \theta(\tau)=\frac{T_{\mathrm{tr}}(\tau)}{T_{\mathrm{rot}}(\tau)}
  \longrightarrow \theta^* .
\]

Therefore the test must show both:

1. total temperature remains positive and cools; and
2. two runs started at `theta0=0.75` and `theta0=1.25` approach the same bounded
   late-time ratio.

For cases with independent DEM values, the late mean must be within 10% of DEM.
The last 30% of each trajectory is used for its mean and drift. The automated
gate requires at most 10% late-window drift, at most 10% disagreement between
the two initial conditions, and at most 10% DEM error.

The blocking gate contains elastic AR 2/3 and the four DEM comparisons at
`alpha=0.95,0.8`, AR 2/3. AR 1.5/2.5 interpolation and AR 1.1/1.2 near-sphere
cases are retained as non-blocking diagnostics. Thus validated domains may move
to excitation without hiding transfer failures in other domains.

## Next Negishi operation

After pulling the repair commit, run one command:

```bash
cd /scratch/negishi/mgbolase/DSMC_V2
git pull
bash hpc/submit_postfit_hcs.sh
```

This performs, in dependency order:

1. a fresh fail-closed check of the 144 existing estimates;
2. artifact construction only (no CTC and no closure refits);
3. a 12-task HCS array to 10 collisions per particle with 1,000 particles;
4. generation of `results/hcs_validation/hcs_theta_attraction.png` and
   `results/hcs_validation/summary.json`.

The local smoke benchmark implies roughly 36 single-core hours for HCS, or
about three to four hours elapsed if all 12 array tasks run concurrently.
Artifact construction uses one 20-core node and is the only other compute-heavy
step. This is deliberately a decision screen: rerun only borderline cases with
2,003 particles, rather than spending that cost on the entire grid up front.

When all jobs finish, inspect:

```bash
jq '{gate_pass, cases}' results/hcs_validation/summary.json
ls -lh models/microscopic_closure_v2/{closure_v2.npz,manifest.json} \
       results/hcs_validation/hcs_theta_attraction.png
```

Proceed to the isotropic HCS excitation responses when `gate_pass` is true. If
only diagnostic cases fail, proceed on the passing AR/alpha domain and record
the exclusions. Do not use USF excitation to compensate for a failed baseline
gate, because that would confound the collision law with the forcing response.
