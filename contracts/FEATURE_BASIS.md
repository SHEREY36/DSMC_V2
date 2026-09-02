# Canonical schema-2.2 invariant basis

`dsmc_v2_contracts.features` is the single implementation imported by CTC
post-processing and DSMC runtime.

Production order:

1. `a2_tr`
2. `a2_rot`
3. `a11`
4. `A_cu = <(cbar.u)^2 - |cbar|^2/3>`
5. `PiPi = (Pi:Pi)/8`
6. `QQ = (Q:Q)/8`
7. `RtRt = (Rt:Rt)/8`, where `Rt = R + Q`
8. `PiQ = (Pi:Q)/4`
9. `PiRt = (Pi:Rt)/4`
10. `QRt = (Q:Rt)/4`
11. `qtr2 = qtr.qtr`
12. `qrot2 = qrot.qrot`
13. `qtr_qrot = qtr.qrot`
14. `W2 = <wbar>.<wbar>`

Diagnostics, not deployed coefficients:

15. `Acw2 = <cbar.wbar>^2`
16. `vx2 = <cbar x wbar>.<cbar x wbar>`

The nondimensional variables use

```text
cbar = C / sqrt(2 Ttr/m),
wbar = omega / sqrt(2 Trot/Iperp).
```

All squared tensor/vector cell quantities use pair U-statistics. Self-products
are subtracted before division by `N(N-1)`, removing the artificial `O(1/N)`
signal from isotropic finite cells. `Acw2` is parity-forbidden at linear order,
and both diagnostics remain outside the production natural-parameter dot
product.

`LEGACY_FEATURE_NAMES` and `legacy_cell_features()` retain the old 16-feature
schema-2.1 order solely for complete historical A/B runs.
