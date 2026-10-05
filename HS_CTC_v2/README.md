# HS_CTC_v2

Fortran collision-trajectory generator for the normalized incoming-flux
measure. The contact model is set by four required environment variables and
written to `metadata_v2.json` (`contact_model_id`):

| model | `CTC_DAMP_VELOCITY` | `CTC_FORCE_LAW` | `CTC_DT_RULE` | `CTC_DT_DIVISOR` |
|---|---|---|---|---|
| **C1** (since 2026-10-05) | `contact` | `hertz` | `energy_bound` | `200` |
| R1 (v1, archived) | `center` | `linear` | `linear_tc` | `50` |

C1 is the contact law of the DEM (LAMMPS `gran/spherocyl/mfix/history ... damp_velocity
contact`): `F = KN DN^{3/2} + CN DN^{1/4} VRN`, `KN = 4/3 E* sqrt(D/4)`,
`CN = sqrt(5) sqrt(KN m*) |ln a|/sqrt(pi^2 + ln^2 a)`, with `VRN` the normal approach speed of
the contact points (`VR + V_ROT`), so the damping never does positive work. The time step is
chosen per encounter from the largest normal speed the pair can reach,
`g_b = sqrt(2 E / m_eff,min)`, so every contact gets at least about `CTC_DT_DIVISOR` steps; it
does not depend on alpha (common random numbers along an alpha line are kept, and the attempt
stream is byte-identical to R1). R1 mode reproduces the v1 binary bit for bit on the same machine.

`contact_diag_v2.bin` (one record per outcome, same order as `outcomes_v2.bin`) holds the
first-contact normal speeds (contact points and centres), the largest overlap, the fewest
steps per contact, and the positive and total damping work; finalization checks that C1
never does positive damping work.

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
