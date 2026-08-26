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
./build/SphCyl ALPHA TTR TROT AR OUTPUT_DIR [SEED] [NSAMPLES] [v2|legacy|both]
PYTHONPATH=../contracts/python python3 scripts/finalize_run.py OUTPUT_DIR
```

`v2` writes every attempted trajectory to `attempts_v2.bin` and every accepted
outcome to `outcomes_v2.bin`. Finalization validates their one-to-one link and
creates the 128-block sufficient-statistics table and `_SUCCESS` marker.

