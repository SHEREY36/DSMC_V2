# HCS non-Gaussian campaign: implementation and production strategy

## Decision

The non-Gaussian measurement and campaign machinery is ready.  The present
candidate artifact is **not** ready for a scientific full-domain campaign:
its baseline surface contains 144 nodes, but its multivariate correction
surface contains only 18 nodes.  The 126 missing correction nodes are now
detected before an expanded HCS allocation is submitted.

The already submitted `submit_usf_candidate_pipeline.sh` remains useful.  It
extends the correction evidence over 18 deliberately selected nodes and runs
a paired USF pilot, but it does not create the final 144-node correction
surface.  Its results should be retained and incorporated into the full-domain
repair rather than duplicated while that pipeline is running.

Safe parallel work is therefore split as follows:

1. Local implementation and validation: complete.
2. Optional engineering HCS-NG pilot at corrected cases: safe with the current
   artifact, but it is not deployment evidence.
3. Full-domain correction build: wait for the submitted extension results so
   they can be reused.
4. Expanded HCS gate: run only after the 144-node correction preflight passes.
5. Scientific HCS-NG domain pilot, map, and tail arms: released only when the
   exact artifact bytes pass that expanded HCS gate.

## Quantities measured

For three translational and two rotational degrees of freedom,

\[
\mathbf c=\frac{\mathbf v-\mathbf U}{\sqrt{2T_{\rm tr}/m}},
\qquad
\mathbf w=\frac{\boldsymbol\omega}{\sqrt{2T_{\rm rot}/I}},
\qquad
x=c^2w^2.
\]

The streaming diagnostic records

\[
a_{20}=\frac{4}{15}\langle c^4\rangle-1,
\qquad
a_{02}=\frac12\langle w^4\rangle-1,
\qquad
a_{11}=\frac23\langle c^2w^2\rangle-1,
\]

along with the spherocylinder-specific orientation correlation

\[
A_{cu}=\left\langle(\mathbf c\!\cdot\!\mathbf u)^2-\frac{c^2}{3}\right\rangle
\]

and the additional translation-rotation quadrupolar correlation

\[
A_{cw}=\left\langle(\mathbf c\!\cdot\!\mathbf w)^2-\frac{c^2w^2}{3}\right\rangle.
\]

The last two quantities are important for transfer from rough spheres/disks to
axisymmetric particles: a spherocylinder has a material axis and hence can
carry correlations that do not exist in the disk calculation.

## Cooling rescaling and collision physics

Long inelastic HCS runs eventually approach floating-point underflow if their
dimensional velocities are allowed to cool indefinitely.  The implemented
similarity transformation is

\[
s=\sqrt{T_{\rm ref}/T},\qquad
\mathbf v\leftarrow s\mathbf v,\quad
\boldsymbol\omega\leftarrow s\boldsymbol\omega,\quad
E_r\leftarrow s^2E_r,\quad
g_{\max}\leftarrow s g_{\max}.
\]

It does not alter the normalized distribution, temperature ratio, cumulants,
orientation axes, NTC acceptance ratios, collision outcome probabilities, or
rotation accumulated per collision.  Scaling `omega`, stored rotational
energy, and the NTC majorant together is essential; the legacy campaign did
not keep all of those quantities synchronized.  Scaled and unscaled arms are
paired in the engineering pilot and must agree within their sampling error
before the rescaled long-time protocol is accepted.

## Statistical production gates

Each realization is an independent one-core Slurm array task.  Physics within
a realization is unchanged; parallelism is only across cases and replicas.
The analysis performs the following checks:

- complete sampling over the declared collision-count window;
- zero negative-energy repairs, energy-axis clamps, and monotonicity repairs;
- late-window stationarity using early-versus-late means;
- integrated autocorrelation time and effective sample size;
- replicate standard errors and deterministic realization-level bootstrap
  confidence intervals;
- scaled/unscaled agreement in the engineering pilot;
- pooled tail-event counts before any exponent is fitted.

Physics and performance verdicts are kept separate.  A trajectory can be a
valid statistical result even when closure overhead misses the 5% deployment
target; only the production verdict requires both.

Histograms use 256 bins.  A tail exponent is not reported until at least 1000
tail particle observations and five occupied tail bins are present.  The fits
use

\[
\phi_c(c)\sim e^{-\gamma_c c},\qquad
\phi_w(w)\sim w^{-\gamma_w},\qquad
\phi_{cw}(x)\sim x^{-\gamma_{cw}}.
\]

Because sampled histograms are radial, the analysis removes the correct
Jacobian before fitting: \(c^2\) for the three-dimensional translational
velocity and \(w\) for the two-dimensional tangent angular velocity.  This
prevents a systematic shift in the fitted power-law exponent.

## Staged designs

| Mode | Physical design | Tasks | Purpose |
|---|---:|---:|---|
| `engineering` | 3 corrected cases, 4 replicas, scaled/unscaled | 24 | timing, stationarity, rescaling equivalence |
| `domain-pilot` | 4 artifact alphas × 7 AR values × 4 replicas | 112 | whole calibrated-axis screening |
| `map` | 10 alphas × 7 AR values × 16 replicas | 1120 | cumulant map, not high-confidence tails |
| `tails` | 9 representative cases × 100 replicas | 900 | high-energy tail statistics |
| `sphere-controls` | 4 alphas × 16 replicas | 64 | separate exact sphere branch |

The map uses \(N=10^4\), \(\tau=500\ldots1500\), and samples every 5
collisions per particle.  Long runs are not assumed to fit the present
24-hour request: the engineering pilot supplies measured seconds per
particle-collision, after which walltimes can be set without guessing.

Negishi has 128 cores per standard CPU node.  The submitter caps simultaneous
one-core realizations at the account's 256-core allocation and uses a strided
array if the site's maximum array size is smaller than the virtual task count.

## Commands after the submitted USF candidate pipeline finishes

The engineering pilot is the only non-Gaussian run permitted with the current
18-node correction surface:

```bash
HCS_NG_MAX_CORES=24 bash hpc/submit_hcs_ng_campaign.sh \
  engineering models/microscopic_closure_v2_candidate/closure_v2.npz engineering_v1
```

After a future artifact has one correction at every baseline node, run the
expanded HCS gate:

```bash
HCS_MODE=full-domain \
HCS_RESULTS=results/hcs_full_domain_candidate_v1 \
HCS_MAX_CORES=256 \
bash hpc/submit_hcs_validation.sh \
  manifests/hcs_full_domain_candidate_v1.csv \
  models/microscopic_closure_v2_full_candidate/closure_v2.npz
```

Only if `full_domain_physics_gate_pass` is true may the domain pilot start:

```bash
HCS_NG_MAX_CORES=256 bash hpc/submit_hcs_ng_campaign.sh \
  domain-pilot \
  models/microscopic_closure_v2_full_candidate/closure_v2.npz \
  domain_pilot_v1 \
  results/hcs_full_domain_candidate_v1/summary.json
```

The `map` and `tails` modes use the same exact-artifact HCS summary argument.
They should be submitted sequentially after inspecting the domain-pilot
stationarity, autocorrelation, runtime, and tail-count results.
