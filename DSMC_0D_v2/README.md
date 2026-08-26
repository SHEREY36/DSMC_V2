# DSMC_0D_v2

Python 0D DSMC consumer for `collision_operator_v2` artifacts. Particle state
contains velocity, laboratory angular velocity, and the spherocylinder axis.

```bash
python3 scripts/run_simulation.py --config config/default.yaml
```

`pair_resolved` is the default. `moment16` is the reduced cell closure. They are
mutually exclusive and never apply two collision-clock corrections.

