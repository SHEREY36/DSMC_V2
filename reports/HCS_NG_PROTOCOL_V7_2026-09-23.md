# HCS non-Gaussian protocol v7 - 2026-09-23

## Decision

Do not advance the completed protocol-v6 engineering result to the long-time
sentinel. All 120 tasks completed and every runtime, memory, NTC, closure-domain,
and performance gate passed, but two of ten numerical-control groups failed.
Both failures were the translational cumulant `a20` at `alpha=0.5, AR=1.35`:

- scaled `dt=0.0025`: `0.026764 +/- 0.000520`;
- unscaled `dt=0.0025`: `0.022446 +/- 0.000791`;
- scaled `dt=0.00125`: `0.023508 +/- 0.000798`.

The scaled-minus-unscaled difference was `0.004318` against a three-SE
resolution of `0.003032`; the scaled-minus-half-step difference was `0.003256`
against `0.002289`. All eight differences had the same sign in the first
comparison and seven of eight in the second. This is not an execution failure
or a relaxation of a tolerance.

## Numerical repair

The orientation-dependent collision measure previously used a first-order Lie
split: process the complete collision batch, apply HCS reheating, and then
advance the axes for one full step. In an unscaled HCS the collision and angular
frequencies cool, so this step error becomes progressively smaller. Similarity
reheating keeps both frequencies stationary, so the same error persists in the
scaled arm.

Protocol v7 uses a symmetric midpoint split:

1. advance every axis for `dt/2` with its incoming angular velocity;
2. evaluate closure features and the orientation-dependent collision measure,
   process collisions, and apply similarity reheating;
3. advance every axis for `dt/2` with its outgoing angular velocity.

This is the time-symmetric Strang composition of free rotation with the
collision-plus-thermostat map. The historical end-step scheme remains the
default for explicit legacy runs, preserving their seeded regression contract.
Every v7 manifest and result records `symmetric_midpoint_v1`, and analysis and
stage preflight reject mixed integrators.

A state-dependent adaptive timestep is deliberately not introduced. In a
rescaled HCS the macroscopic thermal state is stationary, so a temperature- or
collision-rate controller rapidly reduces to a smaller fixed step. Coupling
the timestep to noisy microscopic extrema would add a new sampling dependence
without addressing the operator ordering directly. Fixed `dt` and a paired
half-step arm remain the auditable convergence test.

## Staged allocation

1. Run the 24-task `numerics-pilot` at the sole failed v6 coordinate. It uses
   eight seeds and the scaled `dt=0.0025`, unscaled `dt=0.0025`, and scaled
   `dt=0.00125` arms at the full production particle count and sampling window.
2. Only a passing numerics summary may launch the new 120-task engineering
   campaign. No v6 result is reusable because the orientation integrator
   changed.
3. Only a passing engineering summary may launch the 80-task long-time
   sentinel.
4. The existing 148-task two-sided stability and 370-task production gates
   remain downstream and retain their v6 physical/runtime safeguards.

If the 24-task pilot still resolves a difference, stop. Do not weaken the
control gate: promote the half-step only after adding a still-finer comparison.
