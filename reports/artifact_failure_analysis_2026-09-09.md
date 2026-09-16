# Closure artifact failure analysis — 2026-09-09

## Executive result

The copied Negishi campaign is complete and internally consistent. All 144 raw
CTC shards pass their binary contracts, all 144 estimate JSON files match the
current estimator contract and their source shards, and there are no missing or
duplicate nodes. The build is blocked because 9 of the 144 fitted node models
fail closure QA. Repeating CTC generation would not address this failure.

The nine failures reproduce the earlier failures exactly. This is strong
evidence for a structural model-form problem rather than Monte Carlo noise.

## What “artifact” means

The artifact is the compact, versioned closure that DSMC can execute:

- `closure_v2.npz` stores the calibrated node coordinates, conditional energy
  and angular kernels, quantile sampling tables, collision-measure corrections,
  feature coefficients, uncertainty, and interpolation metadata.
- `manifest.json` records its schema, provenance, hashes, validation results,
  and supported domain.

It is analogous to a trained-model file. Raw CTC records are the training data;
node JSON files are fitted checkpoints; the artifact is the deployable model.
The directory is absent because the builder is fail-closed: it refuses to
package a closure while any required node fails QA.

## Evidence from the copied results

`results/closure_estimates/artifact_grid_validation.json` reports:

- deep binary validation: passed;
- raw shards: 144/144 valid;
- current estimate files: 144/144;
- estimate/source-contract failures: 0;
- closure-QA failures: 9;
- artifact inputs: failed.

All nine nodes fail the held-out model-form test. Two also report a negative
legacy affine-memory diagnostic:

| alpha | theta | AR | held-out gain (nat/event) | fitted bridge memory | other diagnostic |
|---:|---:|---:|---:|---:|---|
| 0.50 | 0.0125 | 1.10 | 0.04312 | 899.9999 | — |
| 0.95 | 0.0125 | 1.35 | 0.02479 | 0.000038 | negative affine `p_exch` |
| 1.00 | 0.0125 | 1.10 | 0.02298 | 791.4204 | — |
| 1.00 | 0.0125 | 1.20 | 0.06476 | 138.6218 | — |
| 1.00 | 0.0125 | 1.35 | 0.07495 | 40.9905 | negative affine `p_exch` |
| 1.00 | 0.0250 | 1.20 | 0.04314 | 108.3533 | — |
| 1.00 | 0.0250 | 1.35 | 0.05705 | 33.1966 | — |
| 1.00 | 0.0500 | 1.20 | 0.02448 | 84.9932 | — |
| 1.00 | 0.0500 | 1.35 | 0.03159 | 27.7627 | — |

The acceptance threshold is 0.02 nat/event. The first node also saturates the
bridge's memory bound, which is a clear numerical symptom rather than a reason
to increase that bound blindly.

## Physical and statistical cause

The Sinkhorn bridge imposes detailed balance on the reduced scalar partition
process at every calibration node. Detailed balance is correct for the full
elastic collision dynamics at equilibrium. It does not generally survive
marginalizing orientation, contact geometry, angular velocity, and the rest of
the collision state at a far-from-equilibrium local ensemble. Near the sphere
limit, torque and translation–rotation exchange weaken, so those hidden
variables retain memory longest. That is exactly where the rejected nodes lie:
very small temperature ratio and low aspect ratio.

The old memory term is linear in the energy fraction, `lambda3*z`. At very small
`theta`, most incoming fractions crowd near zero. Raw powers of `z` then become
poorly scaled and cannot resolve multiplicative changes in a small energy
ratio. Raising the coefficient only makes the optimizer hit a bound; it does
not add the missing curvature.

The reported `p_exch` is not used as a probability by the current DSMC sampler.
It is `1 - slope` from a legacy affine regression used to interpret an older
Bernoulli reset model. A negative value says only that the local affine slope is
slightly above one. The deployed continuous exponential-family kernel consumes
its natural memory coefficients directly. Therefore `p_exch` remains recorded
as an audit diagnostic but no longer vetoes the artifact.

## Targeted repair

Only the nine rejected nodes are changed. Equilibrium nodes at `alpha=1,
theta=1` retain the reversible Sinkhorn bridge and its exact equilibrium
identity. A rejected nonequilibrium node uses

```text
ell = log(z/(1-z))
x   = tanh((ell - weighted_mean(ell))/(2*weighted_sd(ell)))
memory(z) = b1*x + b2*x^2 + b3*x^3
```

and the conditional law

```text
p(z'|z,loss) proportional to Beta(2,2)(z')
  * exp([lambda1 + memory(z) + lambda4*loss]*z' + lambda2*z'^2).
```

The log ratio resolves relative changes near either boundary. Centering and
scaling condition the nonlinear solve. `tanh` bounds every memory coordinate,
so tail events cannot create an unbounded natural-parameter table. A cubic is
the smallest tested family that removed the held-out higher-order signal at
the problematic nodes; a quartic is retained only as a diagnostic competitor.

This does not assert that hidden tensor or velocity-spin invariants are absent.
The homogeneous isotropic CTC ensemble cannot identify corrections whose
ensemble means vanish. Those belong in the existing excitation framework and
must be tested separately. The present failure is already identifiable in the
conditional energy-partition response, so adding unidentifiable features here
would not be justified.

## Compute decision and stop conditions

No new CTC trajectories are requested. The repair workflow performs nine
one-core refits with block bootstrap, a cheap metadata/estimate QA pass, and the
artifact build. The original deep-validation report is preserved; the repaired
estimate QA is written separately.

The artifact build starts only if all nine replacement JSON files pass. It then
checks quantile-table moment accuracy and requires one stable fixed point for
each `(alpha, AR)` slice. A failure at either stage stops the dependency chain
and is new information about the model—not a request to rerun the same job.
