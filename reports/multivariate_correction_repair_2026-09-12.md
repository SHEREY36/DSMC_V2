# Multivariate correction repair and next-stage verdict

## Verdict

The existing 1,296 correction-grid fits are sufficient to build a new
candidate artifact. No excitation rerun is required. All outputs are present,
all 18 physical nodes have full-rank response designs, and every fitted natural
parameter passes the 15% full held-out relative-RMSE criterion.

The candidate is not a production artifact yet. It must pass corrected HCS
dynamics and an independent direct-CTC validation before deployment.

## Repaired model

The old candidate changed only `lambda1`. The repaired response is

\[
\delta\boldsymbol\psi
= B\left(\mathbf X-\mathbf X_0\right),\qquad
\boldsymbol\psi=
(\lambda_1,\lambda_2,\lambda_3,\lambda_4,\eta_1,\eta_2)^\mathsf T.
\]

Central amplitudes \(|\eta|=0.25\) fit (B). Both signs of the
\(|\eta|=0.5\) amplitudes remain held out. The response is released as one
jointly validated Jacobian; deleting individual columns with post-fit t-tests
would bias a full-rank multivariate response.

For the energy kernel, define

\[
a'=\lambda_1+\delta\lambda_1
 +(\lambda_3+\delta\lambda_3)z_{\rm in}
 +(\lambda_4+\delta\lambda_4)\epsilon.
\]

The existing adaptive (a)-axis represents these affine changes. The two
remaining changes of probability-law shape are compiled as central finite
differences of the logit quantile:

\[
\operatorname{logit}Q_{\rm corr}(u)
\simeq \operatorname{logit}Q_0(a',u)
+\delta\lambda_2\,\partial_{\lambda_2}\operatorname{logit}Q_0(a',u)
+\delta\lambda_3\,\left.\partial_{\lambda_3}
  \operatorname{logit}Q_0(a',u)\right|_{a'}.
\]

The second \(\lambda_3\) term is the change in the Sinkhorn bridge potential;
the first occurrence of \(\delta\lambda_3\) in (a') is its memory coupling.
They must both be retained.

The angular corrections add to the matching sufficient statistics:

\[
p(c\mid z)\propto
\exp\left[(\eta_1+\delta\eta_1+\rho z)c
+(\eta_2+\delta\eta_2)P_2(c)\right].
\]

At \(\alpha=1\), detailed balance structurally enforces
\(\delta\lambda_1=\delta\lambda_2=\delta\lambda_4=0\). The memory and angular
responses remain available because they do not create dissipation.

## Offline evidence

- Existing tasks: 1,296/1,296 present; zero execution failures.
- Physical response nodes: 18/18 full rank and well conditioned.
- Full held-out parameter relative RMSE: below 0.15 at every node.
- Global held-out relative RMSE: 0.0232--0.0421 across the six parameters.
- Conditional-energy p95 Wasserstein-1 error:
  - baseline: 0.00821;
  - old `lambda1`-only response: 0.01917;
  - exact six-parameter response: 0.00313;
  - deployable tangent representation: 0.00408.
- Tangent-versus-exact-response p95 Wasserstein-1 error: 0.00213.
- Angular p95 Wasserstein-1 error: 0.0412 baseline versus 0.00931 corrected.

Four extreme points at \(\alpha=1,\theta=0.2\) fail only the optional
higher-order memory-form comparison. Every physical, overlap, projection,
conservation, and elastic sentinel passes. These points mark a boundary warning;
they do not invalidate the central response. The artifact remains restricted to
the calibrated feature hull, and every out-of-domain query is counted.

## Next Negishi stage

`hpc/submit_corrected_artifact.sh` now performs the following chain:

1. recompute the excitation summary with the checked-out code;
2. run the offline multivariate parameter/distribution gate;
3. validate all baseline artifact inputs;
4. rebuild all schema-2.4 node payloads as 128 simultaneous two-core jobs;
5. aggregate a candidate artifact without overwriting the accepted baseline;
6. submit corrected HCS validation after a successful aggregate.

The build uses all 256 available cores. Parallelism changes only scheduling:
the 513-point probability axis, 4097-point density grid, adaptive interpolation
tolerance, and central finite differences are unchanged. Corrected HCS fails
closed on any feature extrapolation above the allowed rate, energy-axis clamp,
negative-energy repair, or quantile-monotonicity repair.
