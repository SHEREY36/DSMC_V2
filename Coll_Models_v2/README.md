# Coll_Models_v2

This package converts keyed CTC attempt/outcome streams into the schema-2.2
BL-compatible variational closure.

The estimator:

- reconstructs `A_perp` and applies inverse-area outcome weights;
- calibrates `A_perp/A0` against all attempt hit flags;
- identifies direct `p_exch` by affine memory regression;
- solves exact energy and angular I-projections;
- measures and gates conditional `z cos(chi)` coupling;
- computes all 14 production invariants from proposal states;
- fits identifiable natural-parameter coefficients from direct ensembles;
- exports `closure_v2.npz` with uncertainties, bounds, masks, quantile tables,
  hashes, and provenance.

It never fits a collision clock or total-loss law and never uses HCS/USF/DEM
validation ensembles for parameter fitting.

```bash
PYTHONPATH=../contracts/python:src \
python3 scripts/estimate_grid.py \
  --runs-root ../results/ctc_closure/sentinel \
  --output ../results/closure_estimates/sentinel --bootstrap 500
```

Artifact construction is intentionally blocked until all supplied node gates
pass. Schema-2.1 baseline shards are accepted through the shared read-only
adapter.
