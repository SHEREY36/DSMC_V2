# Parallel artifact construction on Negishi

## Outcome

Artifact construction is now a fail-closed two-stage Slurm workflow:

1. 144 independent node-precompute work items run as a Slurm array, with at
   most 128 simultaneous two-core tasks (256 cores total).
2. A four-core aggregation job starts only after every array task succeeds. It
   verifies every partial against the current estimate digest, packs the
   adaptive tables, reruns sampler/stability gates, and writes the artifact.

The expensive geometric propensity is cached atomically per raw shard. A
retry reuses completed work rather than repeating it. Existing `*_128.npy`
cache entries remain valid because the threaded implementation is bitwise
identical to the serial block order.

## Why this matches Negishi

Purdue RCAC documents 128 CPU cores and 256 GB per standard Negishi CPU node,
with memory allocated at about 2 GB per requested core. Normal-QoS jobs consume
the group's purchased-core pool. The precompute tasks therefore request two
cores and 4 GB each; the 128-task concurrency cap matches the 256-core account
without oversubscribing memory or BLAS threads.

- Negishi CPU/QoS guidance:
  https://docs.rcac.purdue.edu/userguides/negishi/run_jobs/submit_script/
- Negishi hardware:
  https://docs.rcac.purdue.edu/userguides/negishi/overview/
- Negishi scratch guidance:
  https://docs.rcac.purdue.edu/userguides/negishi/storage/scratch_space/

## Numerical correction

The old builder gave every node a uniform table whose length was determined by
the largest global loss span. At runtime the BL loss covariate is rescaled:

\[
  \ell_{\mathrm{route}}
  = \ell_{\mathrm{BL}}
    \frac{\langle\ell\rangle_{\mathrm{CTC}}}
         {\langle\ell\rangle_{\mathrm{BL}}}.
\]

The new bound applies this exact mapping per node. It then bisects an
`a`-interval only where the midpoint conditional quantile differs from the
runtime's linear interpolation by more than \(2\times10^{-4}\). The 144-node
production grid requires 14,746 rows (maximum 181 at one node), compared with
295,056 rows in the old rectangular layout. The uncompressed quantile payload
drops from about 1.13 GiB to 57.7 MiB.

An independent quarter-point audit over all final intervals found a worst
quantile error of \(6.60\times10^{-5}\). A full packaging dry run loaded all
144 nodes, passed the stability gate, and produced a 54.3 MiB artifact.

## Submission

Cancel the obsolete serial job before updating:

```bash
scancel 43009332
cd /scratch/negishi/mgbolase/DSMC_V2
git pull --ff-only origin closure-v2-repairs
bash hpc/submit_postfit_artifact.sh
```

The submitter reports `precompute_job` and `artifact_job`. Monitor with:

```bash
squeue -A morri353 -o "%.18i %.24j %.2t %.10M %.4C %R"
find results/closure_estimates/artifact_precompute \
  -maxdepth 1 -name 'node_*.npz' | wc -l
```

After the pack job completes:

```bash
jq '{schema_version,n_nodes,n_baseline_nodes,stability_pass,
     energy_table_layout,energy_table_rows,
     maximum_energy_interpolation_error,
     maximum_quantile_moment_error}' \
  models/microscopic_closure_v2/manifest.json
ls -lh models/microscopic_closure_v2/{closure_v2.npz,manifest.json}
```
