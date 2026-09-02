# HS_CTC_v2

Fortran collision-trajectory generator for the normalized incoming-flux
measure. It retains the v1 Hertzian spherocylinder mechanics and deliberately
uses translational relative velocity only in the normal damping law:

```fortran
VREL_CONTACT = VR
```

Build and run:

```bash
make -C build clean all
./build/SphCyl ALPHA TTR TROT AR OUTPUT_DIR [SEED] [NSAMPLES] [v2|legacy|both] [ENSEMBLE_ID]
PYTHONPATH=../contracts/python python3 scripts/finalize_run.py OUTPUT_DIR
```

`v2` writes schema-2.2 data: every attempted trajectory to `attempts_v2.bin` and every accepted
outcome to `outcomes_v2.bin`. `NSAMPLES` is the number of accepted outcomes;
misses do not count against it, but remain in the attempt stream. Finalization
validates the one-to-one hit link, calculates all-attempt energy/score sums,
and creates the 128-block sufficient-statistics table and `_SUCCESS` marker.
The binary record sizes are unchanged; the former reserved header is now
`ensemble_id`. Nonzero excitation ensembles remain hard-gated until the
baseline sentinel and direct-sampler certification have passed.
