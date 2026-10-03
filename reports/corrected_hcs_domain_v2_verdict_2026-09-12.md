# Corrected HCS domain-v2 verdict and model continuation

## Verdict

The correction-enabled candidate has passed the HCS **physics** gate.  It is
stable in every copied trajectory, all six case means are within 3.87% of the
independent DEM targets, and the statistical-domain repair has removed the
false extrapolation alarm without clipping or changing any learned feature.

The aggregate `production_gate_pass=false` is exclusively a runtime-budget
result.  Seven of 36 individual trajectories measured a closure overhead just
above the strict 5% limit.  The campaign-wide mean was 4.594%.  No physical,
numerical-stability, artifact-support, or energy-conservation test failed.

## Copied campaign audit

Source: `results/hcs_corrected_validation_domain_v2/summary.json`, jobs
43106313 and 43106314.

| alpha | AR | mean theta | DEM target | relative error | initial-partition spread | mean-trajectory late drift | replicate CV | physics | production |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0.80 | 2 | 0.91194 | 0.9365 | 2.62% | 3.14% | 5.39% | 2.73% | pass | pass |
| 0.80 | 3 | 0.97649 | 1.0158 | 3.87% | 2.84% | 3.70% | 1.16% | pass | timing only |
| 0.95 | 2 | 0.98986 | 0.9792 | 1.09% | 1.48% | 4.80% | 3.84% | pass | pass |
| 0.95 | 3 | 1.00557 | 0.9962 | 0.94% | 2.31% | 5.08% | 1.13% | pass | timing only |
| 1.00 | 2 | 0.99022 | 1.0000 | 0.98% | 2.30% | 4.70% | 0.42% | pass | pass |
| 1.00 | 3 | 1.01212 | 1.0000 | 1.21% | 0.12% | 0.96% | 1.77% | pass | timing only |

Every one of the 36 runs reported:

- physical out-of-domain fraction exactly zero;
- zero negative-energy repairs;
- zero energy-axis clamps;
- zero quantile-monotonicity repairs.

The non-zero `sampling_excursion_fraction_by_feature` entries are expected
finite-sample U-statistic fluctuations and are deliberately retained as
diagnostics.  They are not physical support violations and they do not alter
the correction input.

All 36 HCS stderr files and the dependent plot stderr file are empty.  The
generated attraction plot is
`results/hcs_corrected_validation_domain_v2/hcs_theta_attraction.png`.

## Runtime repair

The timing profile showed repeated physical-coordinate searches through the
1,440-row artifact table.  This was lookup overhead, not probability-law
evaluation or DSMC collision work.  The runtime now constructs the three
coordinate axes and exact coordinate-to-row map once when the artifact is
loaded.  Each collision reuses them, and the already-computed physical stencil
also determines whether the query is an exact joint-law node.

This changes no model value.  An equivalence audit over 225 exact and
interpolated queries found identical vertex indices and a maximum interpolation
weight difference of exactly zero.  A 10,000-query AR=3 microbenchmark reduced
the coordinate-stencil time from 2.638 s to 1.358 s (48.5%).  In a short
end-to-end profile the closure time fell by 5.5%, sufficient in expectation to
move the former worst systematic case at alpha=1, AR=3 below the 5% budget.

The strict 5% gate has not been weakened.  A fresh timing campaign is required
because wall-clock ratios cannot be retroactively recomputed from the old run.

## What must and must not be recomputed

The candidate artifact, 1,440 precomputed node payloads, 1,296 excitation fits,
and HCS physics evidence remain valid.  They must not be rebuilt.

Only the inexpensive 36-trajectory HCS timing confirmation is needed.  The
tasks are independent and use one core each, so all 36 should run concurrently.
Using more than 36 cores does not speed this particular check and does not
change its accuracy.

```bash
git pull --ff-only origin closure-v2-repairs

bash hpc/submit_corrected_hcs_validation.sh \
  models/microscopic_closure_v2_candidate/closure_v2.npz \
  "" \
  manifests/hcs_corrected_validation_domain_v3.csv \
  results/hcs_corrected_validation_domain_v3
```

After the dependent plot job completes:

```bash
jq '{physics_gate_pass, production_gate_pass, cases}' \
  results/hcs_corrected_validation_domain_v3/summary.json

find logs -maxdepth 1 -type f \
  \( -name 'hcs_gate_*.err' -o -name 'hcs_plot_*.err' \) \
  -size +0c -print

ls -lh results/hcs_corrected_validation_domain_v3/hcs_theta_attraction.png
```

## Next scientific gate

The next model-building operation is an independent fresh-CTC holdout at a
representative inelastic node, followed by comparison of the measured
six-parameter response with the candidate's response Jacobian.  This must use
new random streams, not the shards used to fit the artifact.  It should begin
as a small sentinel and expand only if that sentinel passes.

The old 1,693-row `direct_excitation` design must not be submitted.  Its
hand-written ensemble definitions do not match the currently deployed
rank-normal, one-particle molecular-chaos tilts, and the Fortran executable
intentionally rejects non-zero ensemble IDs.  Submitting it would consume
compute without validating the model that is actually deployed.

The replacement sentinel is implemented by
`hpc/submit_independent_ctc_holdout.sh`.  It does the following, in order:

1. generate two new 200,000-hit baseline CTC shards with a common random stream
   absent from the training campaign: `(alpha,theta,AR)=(0.95,1,2)` and its
   `(1,1,2)` anchor;
2. fit the fresh inelastic baseline, using 20 deterministic geometry workers
   to build the 128-offset propensity cache;
3. evaluate all 18 one-particle excitation families at the two held-out
   boundary amplitudes, for 36 independent validation responses;
4. compare the observed changes in
   `(lambda1,lambda2,lambda3,lambda4,eta1,eta2)` with the frozen artifact
   Jacobian at the actually achieved feature vectors.

The predeclared sentinel criterion is a relative RMSE no larger than 0.20 for
each of the six response parameters over all 36 boundary observations.  This
is deliberately separate from the 10% end-to-end HCS observable criterion:
natural parameters have different scales and the holdout was not used in their
fit.  The JSON retains every observed and predicted response so a failed gate
identifies the exact family and sign to investigate.

The 36 response jobs compute exact point estimates with bootstrap count zero.
Bootstrap resampling changes uncertainty bars, not the fitted point values or
collision physics.  If the sentinel finds a discrepancy, only the implicated
directions should be repeated with bootstrap uncertainty.  This adaptive rule
is what prevents another indiscriminate multi-hour campaign.

The propensity-cache identity has also been hardened for this test.  It now
contains the resolved shard path, seed, byte size, and file modification time.
The former basename-plus-size key could, in principle, alias a fresh holdout
with a training shard having the same directory name and attempt count.  A
regression test constructs exactly that collision and confirms that the two
geometry calculations remain distinct.

After the domain-v3 HCS timing confirmation passes, submit the holdout chain:

```bash
bash hpc/submit_independent_ctc_holdout.sh \
  models/microscopic_closure_v2_candidate/closure_v2.npz \
  results/hcs_corrected_validation_domain_v3/summary.json \
  v1
```

Its last dependent job writes:

```bash
jq . results/closure_estimates/independent_ctc_holdout_v1_validation.json
```

The sentinel uses at most 40 cores for the fresh CTC pair, 20 cores for the
one-time geometric integration, and 36 cores for the response wave.  Those are
the available independent work units; reserving the remaining account cores
would not shorten this gate.  All numerical resolutions and all 200,000 CTC
hits are unchanged by the parallel scheduling.
