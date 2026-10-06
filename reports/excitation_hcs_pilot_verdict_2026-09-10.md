# HCS excitation pilot verdict — 2026-09-10

## Decision

The 96-case HCS excitation pilot succeeded as a **screening and response-fit
experiment**. It is not a deployable correction by itself, and it should not be
repeated merely with a larger bootstrap count.

All 96 virtual ensembles completed; all overlap and physical sentinel checks
passed. The old `deployment_ready=false` value was caused by 58 individual
excited points missing the absolute `lambda1_contribution_precision` target.
That target is useful diagnostic information, but it is not the right test of a
multi-point response regression. More bootstrap resamples of the same collision
sample estimate uncertainty more smoothly; they do not create new independent
collision information.

The revised analysis fits the central amplitudes, \(\eta=\pm0.25\), using
generalised least squares and validates the fit on the unused
\(\eta=\pm0.50\) amplitudes. At one physical node, every response shares the
same baseline estimate, so

\[
  \Sigma_y = \operatorname{diag}(s_1^2,\ldots,s_n^2)
             + s_0^2\mathbf 1\mathbf 1^\mathsf T,
\]

and

\[
  \widehat{\boldsymbol\beta}
  = (X^\mathsf T\Sigma_y^{-1}X)^{-1}
    X^\mathsf T\Sigma_y^{-1}\Delta\boldsymbol\lambda_1.
\]

The deployed coordinate is explicitly centred:

\[
  \Delta\lambda_1
  = \boldsymbol\beta^\mathsf T(\mathbf X-\mathbf X_0),
\]

where \(\mathbf X_0\) is the baseline cell-feature vector. This prevents a
spurious correction at the state about which the response was measured.

## Measured pilot results

- Outputs: 96/96 present and successful.
- Excitation ESS fraction: 0.604 to 0.893; required minimum 0.5.
- Maximum normalized weight share: \(7.78\times10^{-5}\); limit 0.01.
- Four-feature HCS design rank: 4/4 at all six physical nodes.
- Scaled design condition number: 1.45 to 1.47.
- Held-out relative RMSE for \(\lambda_1\): 0.091, 0.132, 0.091,
  and 0.145 at the four inelastic nodes; zero in the elastic limit. The
  predeclared pilot tolerance is 0.15.
- Revised verdict: `screening_pass=true`, `response_fit_ready=true`, and
  `deployment_ready=false` because the full feature basis and dynamical/direct
  validation have not yet been performed.

The inelastic response has the same dominant signs across AR=2 and AR=3:
\(\partial\lambda_1/\partial a_{2,tr}<0\),
\(\partial\lambda_1/\partial a_{2,rot}>0\), and
\(\partial\lambda_1/\partial a_{11}>0\). The \(A_{cu}\) response is smaller
and less consistent, so it must remain uncertainty-gated rather than being
forced into the artifact.

## Staged continuation

1. Run `full-pilot`: 72 virtual ensembles at one representative node, covering
   all 14 production invariants through 18 signed/paired excitation families.
2. If its rank, overlap, and held-out linearity pass, run `correction-grid`:
   1,296 virtual ensembles on 18 physical nodes spanning alpha, theta, and AR.
3. Build a separate candidate artifact. The production artifact is not
   overwritten. Its energy tables are regenerated over the complete calibrated
   correction interval, and cache payloads are cryptographically tied to the
   coefficient fit.
4. Run correction-enabled HCS from both sides of each known attractor. This is
   the first dynamical test of the reduced \(\lambda_1\)-only correction.
5. Only after that passes, implement and run selected fresh direct-CTC
   excitations. The current Fortran CTC initializer accepts only baseline
   `ensemble_id=0`; importance-reweighted virtual excitation does not replace
   this independent validation.
6. The 10,368-task `production-grid` is prepared but deliberately gated behind
   the smaller correction grid. It should not be submitted before the candidate
   model passes dynamics and direct validation.

Parallel execution changes only scheduling. Each virtual fit keeps the same
collision records, estimator, propensity resolution, and deterministic seed.
Independent fits are distributed across cores; no samples are thinned and no
numerical tolerances are relaxed.
