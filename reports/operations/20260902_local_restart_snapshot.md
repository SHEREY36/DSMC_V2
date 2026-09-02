# CTC restart snapshot — 2026-09-02

- Repository commit at inspection: `bae4562ffc8ee72048a4251cd9cecabf7d685a40`.
- Working tree was clean and `master` was four commits ahead of `origin/master`
  before implementation began.
- The workstation has no `squeue`, `sacct`, or `scancel` client.
- The short host `negishi` failed DNS resolution. The configured host is
  `negishi.rcac.purdue.edu`, but remote access was not approved, so no Slurm
  job has been cancelled from this environment.
- No result directory was deleted, moved, or overwritten.

Run the following from the repository checkout on a Negishi login node:

```bash
bash hpc/cancel_ctc_jobs.sh --cancel
```

That command snapshots queue/accounting state, the commit SHA, all `_SUCCESS`
markers, all run metadata, and incomplete-directory paths before cancelling
only array parents whose job names match this repository's CTC names.
