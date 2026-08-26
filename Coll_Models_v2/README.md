# Coll_Models_v2

Standalone conversion of finalized CTC records into DSMC collision operators.
It does not read HCS, USF, Fourier, DEM, or LAMMPS ensemble results.

```bash
python3 scripts/estimate_node.py RUN [RUN_SHARD ...] --output node.json
python3 scripts/estimate_grid.py --runs-root ../results/ctc --output ../coefficients
python3 scripts/build_artifact.py --runs-root ../results/ctc \
  --output ../models/collision_operator_v2
```

The output separates pair/cell collision clock, joint retained-energy outcome,
loss routing, and direction-only VSS rank-2 scattering.

