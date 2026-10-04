# CTC-learned collision closure for DSMC of inelastic spherocylinders (final_v1)

This folder is the complete collision model. Every law in it was fitted from exact
collision dynamics (CTC encounters at Maxwellian states); no DEM data entered any fit.

| file | contents |
|---|---|
| `closure_v2.npz` | 627 nodes: exchange kernels $p(z'\mid z,\varepsilon)$ (quantile tables, Sinkhorn potentials, natural parameters), spin-enhancement curves $g(\Xi)$, node coordinates |
| `encounter_cross_section.json` | measured dynamic factor $D(\theta,\mathrm{AR})$ of $\sigma_c=\bar A_\perp D$, and contacts per encounter $\langle k\rangle(\alpha,\theta,\mathrm{AR})$ |
| `loss_memory.json` | loss rates $c_t$, $c_r$ and the mean per-encounter loss at every node: $\mathbb E[\varepsilon\mid z]=c_t z+c_r(1-z)$ |
| `angular_memory.json` | scattering law $p(c\mid z,z')\propto\exp[(\eta_1+\xi z'+\rho_1 z)c+(\eta_2+\zeta z'+\rho_2 z)P_2(c)]$ at every node |
| `manifest.json` | artifact build record, including the HCS stability check of every $(\alpha,\mathrm{AR})$ |
| `model_card.json` | required tables, grid, laws, provenance and SHA-256 of every file |
| `model.yaml` | a ready DSMC_0D_v2 configuration with paths relative to this folder |
| `SHA256SUMS` | checksums (`sha256sum -c SHA256SUMS`) |

**Grid.** $\alpha$ = 0.50, 0.55, …, 0.95, 1; AR = 1.1, 1.2, 1.35, 1.5, 2, 2.5, 3;
$\theta=T_{tr}/T_{rot}$ = 0.0125–0.2 (six values), 0.35, 0.5, 0.7, 1, 2 for AR ≤ 1.35 and 0.2,
0.35, 0.5, 0.7, 1, 2 for AR ≥ 1.5. Between nodes the laws are mixed on a physical stencil; the
model is validated for AR 1.1–3 and $\alpha$ 0.5–1.

**Running it.** The runtime is the `DSMC_0D_v2` package of the DSMC_V2 repository (commit
`e0bfbcc` or later). From this folder:

```bash
export PYTHONPATH=<repo>/contracts/python:<repo>/Coll_Models_v2/src:<repo>/DSMC_0D_v2/src
python -c "import yaml; from dsmc_v2.simulation import run_simulation; \
c = yaml.safe_load(open('model.yaml')); c['flow'] = {'mode': 'hcs', 'shear_rate': 0.0}; \
c['preprocessing']['model_root'] = '<repo>/DSMC_0D_v2/models'; \
print(run_simulation(c, 42, 'hcs.txt')['termination_reason'])"
```

`preprocessing.model_root` points at the repository's legacy BL tables (the Beta law's
normalisation, which cancels from the loss draw). Edit `particle.AR`, `system.alpha`, the initial temperatures and `flow` (`hcs` or `usf` with a
shear rate) in `model.yaml`. The simulation refuses to run `closure_v2.npz` without the three
tables (`model_card.json`); in particular the loss scale comes from `loss_memory.json`, because
the artifact's `energy_mean_loss` holds energy-weighted node means used only for diagnostics.

**Formulation.** `reports/DSMC_CLOSURE_FORMULATION_FINAL.md` in the repository.
