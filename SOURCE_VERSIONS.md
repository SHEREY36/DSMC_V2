# Version-1 provenance for the conservative v2.1 baseline

The original repositories remain independent and unchanged.

| v2 package | v1 repository | imported commit |
|---|---|---|
| `HS_CTC_v2` | `HS_CTC` | `7d60e34` |
| `Coll_Models_v2` | `Coll_Models` | `510f0f2` |
| `DSMC_0D_v2` | `DSMC_0D` | `cd6340b` |

Version 2.1 mechanically imports the following authoritative v1 runtime files
from `DSMC_0D@cd6340b`:

| v2 file | SHA-256 |
|---|---|
| `DSMC_0D_v2/src/dsmc_v2/ntc.py` | `f75f2ac18d6e5bf938f3bb8401fac887593e45cc5df6126ef1e9a43bd7e4ed8d` |
| `DSMC_0D_v2/src/dsmc_v2/gmm_energy.py` | `2cff47fa34e9a1d9ea2b9d174af12acf5ec3f13c82f3dad76cb62a19ffb96535` |
| `DSMC_0D_v2/src/dsmc_v2/pressure.py` | `a8a0bdb21bf17510c589175ddb1b55f237370d7555002d8b07ddd7fedda0e17c` |

The unchanged v1 scalar-energy inputs are version-controlled because they are
required by both the routing denominator and runtime:

| artifact | SHA-256 |
|---|---|
| `gamma_max_table.json` | `bf12d6109858bca976bd6fd102dbcf7edc2fc85ff61377100c44ed3629a735e5` |
| `one_hit_table.json` | `830464da31d7d2bfaf8718e2a7a8ce1fe50e5ccf924bfd72976651fd54592e0f` |
| `gmm_cond_AR11.npz` | `07cd5cd061ce98bd3383ef843d6e4904a3568e9f3c309c4adc56cffcbd2f09ce` |
| `gmm_cond_AR125.npz` | `9b9212d2f9909590c1f0257c717869b9c0da45f536c946c93ffb5455e55a959e` |
| `gmm_cond_AR15.npz` | `014c63eb812f79230b379fffb9a90a722ee6f8d6a0233c5fccc28cefed4a365e` |
| `gmm_cond_AR20.npz` | `7aee4c6f81717c6116c5e9d58bff38b286d3d97068e0ab4133b05a522f9f6814` |
| `gmm_cond_AR25.npz` | `9c108c08de30903ec909bbef9f4e2e3b7200db2a91ca876359483958fa61aafa` |
| `gmm_cond_AR30.npz` | `804538b07b2601be972f980bb1edc7c70d1421ef638d9b45624e5a6b08253c66` |

No v1 DEM-calibrated routing coefficient or `p_eta` artifact is imported into
the new production closure. Generated data, notebooks, figures, and result
directories are not copied.
