# DSMC_0D_v2

The `variational_v2` runtime keeps the v1 NTC clock, fixed cross-section, and
scalar BL loss while replacing the legacy conditional-GMM/routing composition
and VSS angle.

```yaml
microscopic_closure:
  routing: variational_v2
  angular: variational_v2
  artifact: models/microscopic_closure_v2/closure_v2.npz
  invariant_corrections: true
```

The exchange gate stores and uses `p_exch` directly. A closed gate preserves
the incoming translational and individual rotational partitions; an open gate
draws the exact tilted reset law and uniform rotational sub-split. The angular
draw matches two Legendre moments and uses uniform azimuth. All new modal
energies are positive by construction, and no repair clipping exists in this
branch.

The frozen complete legacy mode remains:

```yaml
microscopic_closure:
  routing: legacy_rank0
  angular: legacy
```

Run from the monorepo root:

```bash
PYTHONPATH=contracts/python:Coll_Models_v2/src:DSMC_0D_v2/src \
hpc/python.sh DSMC_0D_v2/scripts/run_simulation.py \
  --config DSMC_0D_v2/config/default.yaml
```

Diagnostics report closure overhead, feature-domain frequency, and the number
of negative-energy repairs (required to remain exactly zero).
