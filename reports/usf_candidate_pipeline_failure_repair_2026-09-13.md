# USF candidate pipeline failure and repair — 2026-09-13

## Verdict

The Negishi simulation work completed successfully. All 1,296 USF-extension
excitation outputs exist, report `excitation_status=pass`, and pass screening.
The pipeline stopped at job `43114296` because its fail-closed response QA
returned nonzero. Slurm then cancelled every `afterok` dependency, so no
candidate artifact, HCS summary, fresh-CTC rescore, or USF pilot was created.

The existing excitation results are reusable. They must not be rerun.

## Exact failure

Only one natural-parameter response caused the node-level gate to fail:

- physical node: \((\alpha,\theta,AR)=(0.8,2.0,1.5)\);
- parameter: \(\eta_2\);
- maximum measured shift: 1.843 standard errors;
- held-out relative RMSE: 0.199, above the 0.15 material-response limit;
- held-out absolute RMSE: 0.00510.

This is an unresolved response: the observed change never reaches the declared
three-standard-error materiality threshold. A relative error formed by dividing
by this near-zero signal is large even though its absolute error is small. The
old artifact fitter nevertheless deployed all 14 coefficients for every row,
including this noise-dominated row, and the summary treated its relative error
as a model blocker.

A second contract mismatch affected distribution QA. The Jacobian is fitted on
the central \(|\eta|=0.25\) excitations, while \(|\eta|=0.5\) is held out. The old
gate demanded that the first-order runtime tangent remain accurate over the
held-out amplitude and the artifact builder incorrectly enlarged runtime
feature support with those held-out points.

## Repair

1. A natural-parameter row is deployed only when the response is resolved from
   zero by at least three standard errors. An unresolved row is set exactly to
   zero; this removes noise rather than discarding measured physics.
2. Material rows remain joint 14-feature Jacobian rows and must still pass the
   same 0.15 held-out relative-RMSE limit.
3. Runtime feature limits are now local to each coefficient node and use only
   baseline plus fitted \(|\eta|\leq0.25\) observations. The runtime continues to
   fail closed outside that support.
4. The \(|\eta|=0.5\) observations remain independent held-out evidence for the
   exact natural-parameter response; they no longer masquerade as support for a
   first-order tangent fitted at half that amplitude.
5. A resume submitter reuses the 1,296 completed fits, repeats only cheap QA,
   and then rebuilds and validates the exact candidate bytes.

No acceptance threshold was relaxed.

## Recomputed local evidence

The repaired audit of the copied Negishi outputs reports:

- 1,296 tasks, zero missing/failed tasks;
- screening, held-out sentinel, response-fit, and candidate-artifact gates pass;
- global held-out relative RMSE: 0.0205 (`lambda1`), 0.0344 (`lambda2`),
  0.0195 (`lambda3`), 0.0176 (`lambda4`), 0.0260 (`eta1`), 0.0429 (`eta2`);
- central-region tangent-to-exact Wasserstein-1 p95: 0.00112 (limit 0.005);
- central energy error p95: 0.00196 corrected versus 0.00479 baseline;
- central angular error p95: 0.00205 corrected versus 0.01650 baseline;
- the exact multivariate response also improves the independent
  \(|\eta|=0.5\) energy and angular comparisons.

These results authorize an artifact rebuild, not deployment. The rebuilt bytes
must still pass HCS, independent fresh-CTC rescoring, and the paired USF pilot.

## Non-Gaussian campaign

The production non-Gaussian HCS campaign must wait. It must use an artifact
whose corrected dynamics have passed HCS over its claimed domain; the missing
candidate artifact has obviously not done that. The campaign implementation
and tests can be committed now, but no production/map/tail submission is
scientifically authorized by this repair. A small engineering timing pilot or
exact-sphere control is not physics validation of the candidate and is not the
next priority while the candidate pipeline is pending.
