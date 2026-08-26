# Canonical 16-feature basis

The shared implementation is `dsmc_v2_contracts.features`. Both estimation and
runtime import it; neither package keeps a private copy.

1. `a2_tr`
2. `a2_rot`
3. `a11`
4. `PiPi = (Pi:Pi)/8`
5. `RR = (R:R)/8`
6. `QQ = (Q:Q)/8`
7. `PiR = (Pi:R)/4`
8. `PiQ = (Pi:Q)/4`
9. `RQ = (R:Q)/4`
10. `qtrqtr`
11. `qrotqrot`
12. `qtrqrot`
13. `a3_tr`
14. `a3_rot`
15. `a21`
16. `a12`

Quadratic tensor/vector cell quantities use pair U-statistics. This subtracts
self-products before division by `N*(N-1)` and removes the artificial `O(1/N)`
signal in an isotropic finite cell.

