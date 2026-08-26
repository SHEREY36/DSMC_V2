# Coll_Models_v2

This package reads finalized CTC attempt/outcome streams and exports only the
three conservative microscopic closures:

- `routing16_v2.json` for the BL-compatible translational loss split;
- `vss_rank2_v2.json` for direction-only rank-2 scattering;
- `rotational_direction_v2.npz` for paired post-spin directions.

It does not fit a collision clock, replace the conditional GMM, correct total
loss, or read DEM ensemble results.

```bash
python3 scripts/estimate_grid.py \
  --runs-root ../results/ctc --output ../coefficients --bootstrap 2000 \
  --gamma-max-table models/legacy_bl/gamma_max_table.json \
  --one-hit-table models/legacy_bl/one_hit_table.json

python3 scripts/build_artifact.py \
  --runs-root ../results/ctc --output ../models/microscopic_closure_v2 \
  --gamma-max-table models/legacy_bl/gamma_max_table.json \
  --one-hit-table models/legacy_bl/one_hit_table.json
```

The copied JSON tables define the unchanged v1 BL denominator. They are
read-only inputs, not refitted outputs.

