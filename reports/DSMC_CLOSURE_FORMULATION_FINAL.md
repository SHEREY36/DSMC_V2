# DSMC for inelastic spherocylinders with a CTC-learned collision closure: final formulation

Model folder: `models/microscopic_closure_v2_final_v1/` (artifact `closure_v2.npz` and three
tables; `model_card.json` lists them with checksums). Code: branch `closure-v2-repairs`,
commit `e0bfbcc` or later. Configuration: `DSMC_0D_v2/config/encounter_unit_model_final.yaml`,
or the folder-relative `model.yaml` inside the model folder.

## 1. Collision event and reduced variables

One DSMC event is one **encounter** as resolved by the exact collision solver (CTC): a
force-free approach, the first contact, any further contacts while the rods rotate, and the
final separation. For a colliding pair with relative velocity $\mathbf g$, spins
$\boldsymbol\omega_{1,2}$ and axes $\mathbf u_{1,2}$,

$$E_{tr}=\tfrac m4 g^2,\qquad E_{rot}=\tfrac I2(\omega_1^2+\omega_2^2),\qquad E=E_{tr}+E_{rot},\qquad z=E_{tr}/E .$$

The encounter removes the fraction $\varepsilon$ of $E$, leaves the share $z'$ of the remaining
energy $E_f=(1-\varepsilon)E$ in translation, and turns $\hat{\mathbf g}$ by the cosine
$c=\hat{\mathbf g}\cdot\hat{\mathbf g}'$. The closure is the conditional law of
$(\varepsilon,z',c)$ given the incoming pair, factorised in the order the encounter draws them:

$$K(\varepsilon,z',c\mid z)=p(\varepsilon\mid z)\;p(z'\mid z,\varepsilon)\;p(c\mid z,z').$$

Each factor conditions only on variables fixed before it, so the factorisation is exact (chain
rule) and the three laws are fitted independently from the same CTC encounters.

## 2. Nodes

Every law is fitted at Maxwellian nodes $k=(\alpha_k,\theta_k,\mathrm{AR}_k)$ from about
200,000 CTC encounters each, with $\theta=T_{tr}/T_{rot}$:

| | values |
|---|---|
| $\alpha$ | 0.50, 0.55, …, 0.95, 1 |
| $\theta$, AR ≤ 1.35 | 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.7, 1, 2 |
| $\theta$, AR ≥ 1.5 | 0.2, 0.35, 0.5, 0.7, 1, 2 |
| AR | 1.1, 1.2, 1.35, 1.5, 2, 2.5, 3 |

627 nodes in total. The planes of one $(\theta,\mathrm{AR})$ coordinate share one CTC attempt
stream, so differences between $\alpha$ planes carry no sampling noise from the incoming pairs.
No DEM data enters any fit.

## 3. The laws

**Clock.** The encounter cross-section is measured as $\sigma_{enc}=A_{prop}N_{hit}/N_{att}$
(impact points uniform on a square of side $2b_{max}$, $b_{max}=1.01(L+D)$) and written as

$$\sigma_c(\theta,\mathrm{AR})=\bar A_\perp(\mathrm{AR})\,D(\theta,\mathrm{AR}),\qquad
\bar A_\perp=\pi\big(d^2+dL+\tfrac{L^2}{8}\big),$$

the exact isotropic mean projected excluded area times a measured dynamic factor $D$. NTC draws
candidates at $\sigma_c$ times the majorant of the second-stage acceptance.

**Pair selection.** A candidate pair is kept with probability
$A_\perp(\mathbf u_1,\mathbf u_2,\hat{\mathbf g})\,g(\Xi)/(A_{sup}\,g_{max})$, with the exact
projected excluded area

$$A_\perp=\pi d^2+2dL\big(|\mathbf u_1\times\hat{\mathbf g}|+|\mathbf u_2\times\hat{\mathbf g}|\big)+L^2|\hat{\mathbf g}\cdot(\mathbf u_1\times\mathbf u_2)|$$

and a spin enhancement $g(\Xi)$, $\Xi=(|\boldsymbol\omega_1|+|\boldsymbol\omega_2|)(L+D)/g$,
tabulated from the exact kinematic encounter propensity of every CTC proposal.

**Loss.** The CTC conditional mean is linear in the incoming share,

$$\mathbb E[\varepsilon\mid z]=c_t\,z+c_r\,(1-z),$$

fitted by least squares over each node's encounters: the loss rates of translational and
rotational energy. Near the sphere $c_r\to0$ (a smooth sphere loses only translational
energy); for long rods $c_t\approx c_r$. The draw is

$$\varepsilon=\mathbb E[\varepsilon\mid z]\;\frac{B}{\langle B\rangle},\qquad B\sim\mathrm{Beta}(1.21,3.67),$$

so the node's per-encounter mean and its dependence on $z$ come from CTC, and $B$ supplies the
shape. Without the $z$ dependence the dissipation $\langle\varepsilon E\rangle$ is off by the
covariance of $\varepsilon$ and $E$ wherever $\theta\neq1$ (2 % at AR 1.5 in USF, 24 % at AR 1.1,
$\theta=0.2$), and the loss is wrongly split between the modes.

**Exchange.** The outgoing share follows the conditional exponential family

$$p(z'\mid z,\varepsilon)\propto 6z'(1-z')\,h(z)\,h(z')\,
\exp\!\big[\lambda_3 zz'+\lambda_1 z'+\lambda_2 z'^2+\lambda_4\varepsilon z'\big],$$

where $h$ is the Sinkhorn potential that makes the kernel reversible with respect to the elastic
equipartition law, so equipartition is exact at $\alpha=1$. At 25 nodes (24 of them near the sphere, AR ≤ 1.35, at
$\theta\le0.1$; the 25th at AR 2.5, $\theta=0.2$) the memory $\lambda_3 z$ is replaced by a cubic in
$x=\tanh[(\mathrm{logit}\,z-\mu)/2s]$. The parameters are the **energy-weighted I-projection**

$$\hat{\boldsymbol\lambda}=\arg\max_{\boldsymbol\lambda}\sum_i E_{f,i}\,\log p(z'_i\mid z_i,\varepsilon_i;\boldsymbol\lambda),$$

whose stationarity conditions are $\sum_i E_{f,i}\,[T(z'_i)-\mathbb E_p(T\mid z_i,\varepsilon_i)]=0$
for each sufficient statistic $T$. Because the modal energy changes are
$\Delta E_{tr}=E_f z'-Ez$ and $\Delta E_{rot}=E_f(1-z')-E(1-z)$, these identities make every
node reproduce the energy each mode gains or loses, at any $\theta$. An unweighted fit does so
only at $\theta=1$, where $z$ and $E$ are independent. On the fitted near-sphere nodes the
unweighted kernel had the HCS rate $d\ln\theta/d\tau$ wrong in sign at AR 1.1, $\theta=0.025$,
and 14–49 % off at $\theta=0.05$–0.2; the energy-weighted fit reproduces CTC exactly.

**Scattering angle.**

$$p(c\mid z,z')\propto\exp\!\big[(\eta_1+\xi z'+\rho_1 z)\,c+(\eta_2+\zeta z'+\rho_2 z)\,P_2(c)\big],$$

with the azimuth uniform about $\hat{\mathbf g}$, fitted as a MAP I-projection with weight $E_f$
(the identity for $T=P_2(c)$, $h=z'$ is the traceless stress production of one encounter). The
outgoing relative velocity is $\mathbf g'=2\sqrt{z'E_f/m}\,\hat{\mathbf g}'$.

**Spins.** $E'_{rot}=(1-z')E_f$ is split between the particles with a uniform fraction, and each
spin is given a uniform direction perpendicular to its axis.

**Limits.** $\alpha=1$ uses the exact elastic block (Borgnakke–Larsen with $Z_r=5/3$, exact
because the elastic law is the equipartition law); AR = 1 uses the exact smooth-sphere collision.

## 4. Between nodes

For a cell state $(\alpha,\theta,\mathrm{AR})$, $\theta$ being the cell's instantaneous
$T_{tr}/T_{rot}$, the stencil is the tensor product of linear weights in $\alpha$, $\theta$ and
AR (a value on a grid line takes that line alone), with barycentric Delaunay weights where a box
vertex is missing. The energy law is the Wasserstein average of the node laws (node quantiles
averaged at one uniform deviate), the angular law is their mixture, the loss mean is the mixture
of node conditional means, and $D(\theta,\mathrm{AR})$ is interpolated in $\log\theta$ and AR.
Every draw stays a draw from a physical law.

## 5. One DSMC step

1. Free rotation of the axes over $\Delta t/2$ (exact Rodrigues rotation); USF shear of the
   velocities.
2. Cell state $T_{tr}$, $T_{rot}$, $\theta$; every 0.05 collisions per particle, the stencil and
   the interpolated scalar tables.
3. NTC candidates at $\sigma_c(\theta,\mathrm{AR})$; speed screen $\propto g$; second-stage
   acceptance $A_\perp g(\Xi)$.
4. For each accepted pair: $z$; $\varepsilon$ from the loss law; $z'$ from the exchange law;
   $c$ from the angular law; new velocities about the unchanged centre of mass; new spins.
5. Free rotation over the second $\Delta t/2$.

The cost is $O(N\tau)$ with a fixed amount of work per encounter.

## 6. Diagnostics used during development (not part of the model)

The **production test** compares the model's collisional productions with exact CTC on
identical replayed pairs: dissipation $D$, rotational production $\Lambda_{rot}$, and the shear
and normal-stress productions $\Lambda_{xy}$, $\Lambda_{N_1}$. A rotational residual
$r_{rot}=\delta\Lambda_{rot}/D$ moves the steady $\theta$ by
$\delta\theta/\theta\approx-\kappa\,r_{rot}$, $\kappa=D/(\theta\,\partial_\theta\Lambda_{rot})$.
The HCS form compares $d\ln\theta/d\tau$ on Maxwellian node pairs. It located the event-unit
error, the $\alpha$-interpolation error, the loss–partition dependence and the unweighted-fit
error; it is a test and does not enter the model.

## 7. Validation

Independent DEM (LAMMPS, $\phi=0.01$, corrected Hertz damping; never used in fitting). Filled in
from `results/validation/usf/usf_encounter_<tag>/` and `results/validation/hcs/hcs_encounter_<tag>/`.
