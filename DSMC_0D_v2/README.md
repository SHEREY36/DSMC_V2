# DSMC_0D_v2

The runtime keeps the v1 NTC clock, cross-section, conditional GMM, BL total
loss, reservoir handling, HCS output, and USF pressure accumulation. Its only
new production switches are:

```yaml
microscopic_closure:
  routing: ctc_moment16  # or legacy_rank0
  angular: ctc_vss_rank2 # or legacy
```

Routing changes only the translational/rotational split of an unchanged loss
draw. VSS changes only the outgoing relative direction at an unchanged speed.

From the monorepo root:

```bash
PYTHONPATH=contracts/python:Coll_Models_v2/src:DSMC_0D_v2/src \
python3 DSMC_0D_v2/scripts/run_simulation.py \
  --config DSMC_0D_v2/config/default.yaml
```

The unchanged GMM artifacts support `AR=1.1, 1.25, 1.5, 2.0, 2.5, 3.0`.
`AR=1` uses the analytic sphere bypass. No AR extrapolation is permitted.

