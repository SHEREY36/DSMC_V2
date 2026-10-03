# Fresh-CTC holdout verdict and route to USF

Date: 2026-09-12

## Verdict

The independent fresh-trajectory response validation passed.  It is valid
evidence that the fitted invariant-correction Jacobian predicts collision-law
responses on CTC trajectories that were not used to fit the candidate.

This result clears the *statistical response* gate; it does not by itself make
the artifact deployable for the requested uniform-shear-flow (USF) domain.
The current correction table covers only

\[
0.8\leq\alpha\leq1,\qquad 0.2\leq\theta\leq2,
\qquad 2\leq \mathrm{AR}\leq3.
\]

The baseline collision law covers the wider physical surface, but a corrected
USF query at \(\alpha=0.5\) or \(\mathrm{AR}=1.5\) is correctly rejected as
out of the correction model's calibrated hull.  Launching the desired USF grid
before closing this gap would therefore spend compute on a known model-support
failure.

## Independent holdout evidence

Source:
`results/closure_estimates/independent_ctc_holdout_v1_validation.json`.

- Expected/valid/failed responses: 36 / 36 / 0.
- Minimum effective-sample-size fraction: 0.603551.
- Maximum normalized importance-weight share: 0.0001361.
- Response verdict: pass.
- Overall independent-validation verdict: pass.

| Parameter | Correlation | Relative RMSE | Maximum absolute error |
|---|---:|---:|---:|
| \(\eta_1\) | 0.9691 | 0.0405 | 0.03184 |
| \(\eta_2\) | 0.9070 | 0.0797 | 0.02290 |
| \(\lambda_1\) | 0.9624 | 0.0602 | 0.23129 |
| \(\lambda_2\) | 0.9569 | 0.0457 | 0.25808 |
| \(\lambda_3\) | 0.9733 | 0.0383 | 0.10956 |
| \(\lambda_4\) | 0.9633 | 0.0436 | 20.53477 |

The absolute error for \(\lambda_4\) is numerically large only because that
parameter's observed response scale is about 125.88; its scale-free error is
4.36%.  The worst scale-free result is \(\eta_2\) at 7.97%, well below the
predeclared 20% holdout threshold.

The fresh CTC generation produced 200,000 accepted collisions for each of the
inelastic response node and its elastic anchor, from 447,761 attempts each.
All relevant Slurm error logs are empty.  The 36 boundary excitation outputs
all completed and the holdout QA job exited successfully.

## Minimal support extension

No new CTC trajectories are needed.  Existing baseline shards already cover
the required boundary planes.  The minimum additional coefficient design is

\[
\begin{split}
&\alpha=0.5,\quad \theta\in\{0.2,1,2\},
\quad \mathrm{AR}\in\{2,3\},\\
&\mathrm{AR}=1.5,\quad
\alpha\in\{0.5,0.8,0.95,1\},\quad
\theta\in\{0.2,1,2\}.
\end{split}
\]

These are 18 disjoint physical nodes.  With 18 feature families and four
signed amplitudes, they produce 1,296 virtual response fits.  Together with
the existing 18 coefficient nodes, their convex hull covers

\[
0.5\leq\alpha\leq1,qquad 0.2\leq\theta\leq2,
\qquad 1.5\leq\mathrm{AR}\leq3.
\]

Intermediate restitution values and \(\mathrm{AR}=2.5\) are interpolation
tests, not extra fitted planes.

## Accuracy-preserving parallel workflow

The prior long runs were dominated by a deterministic 128-offset geometric
encounter integral.  Starting all 72 fits at a node before its cache exists
can make them race and repeat that same integral.  The new workflow is:

1. Compute one propensity cache per physical node: 18 jobs, 12 cores each.
2. After all caches pass, run 1,296 one-core response fits, up to 256 at once.
3. Apply the response-linearity and offline distribution gates.
4. Recompute all 144 artifact sampler payloads because the correction digest
   has changed, using 128 two-core work queues at once.
5. Pack a new, separate candidate artifact; the existing production artifact
   is not overwritten.
6. Rerun corrected HCS and rescore the already-fresh CTC holdout against the
   exact new artifact bytes.  Fresh CTC trajectories are not regenerated.
7. Only if both gates pass, run a paired corrected/uncorrected USF pilot.

The new gate records and verifies the artifact SHA-256 in HCS, fresh-CTC, and
USF outputs, so a passing JSON from an older `.npz` cannot release a different
candidate accidentally.

The cache-first schedule does not reduce the offset count, bootstrap count,
particle count, simulated collision duration, or artifact table resolution.
It removes duplicate work only.

## USF observables and decision

The v2 USF runner now records translational/rotational temperatures, kinetic
and collisional pressure, and the nematic tensor

\[
\mathbf Q=\langle\mathbf u\mathbf u\rangle-\frac{1}{3}\mathbf I.
\]

Particle axes use a vectorized exact Rodrigues rotation for
\(\dot{\mathbf u}=\boldsymbol\omega\times\mathbf u\).  The homogeneous shear
streaming update is the exact constant-step map here because the simple-shear
velocity-gradient matrix is nilpotent.

The paired pilot uses the same physical cases and seeds with corrections off
and on.  It asks both whether the corrected model is stable and within the
predeclared 10% aggregate accuracy target, and whether the learned correction
actually reduces the held-out LAMMPS particle-simulation stress error.  The full 160-run USF grid is
not submitted automatically; it is released only after inspecting the pilot.
