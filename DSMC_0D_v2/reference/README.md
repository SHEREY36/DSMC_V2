# Uniform-shear validation reference

`usf_dem_reference.csv` contains the dilute particle-simulation steady-USF
values used in the existing thesis Figure 5 comparison.  The archived source
labels these as its `LAMMPS` columns; they were copied from
`../DSMC_0D/figures/usf_chapter/usf_chapter_steady_data.csv`.  No
previous DSMC prediction or hand-edited DSMC value is used by the v2 gate.

The four pressure components are reduced by `n T_tr`.  `theta` is
`T_tr/T_rot`.  The table covers the simulated inelastic range
`alpha=0.50,...,0.95`; the plotted elastic point at `alpha=1` was an imposed
boundary in the earlier figure and is deliberately excluded from validation.

The archived table identifies these columns as LAMMPS particle data.  Before final
publication, this compact table should be checked against the original raw DEM
directories or regenerated raw exports.  That provenance check is separate
from the frozen-model USF gate and must not alter predictions after seeing
their errors.
