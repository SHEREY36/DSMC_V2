# Closure-v2 copied-campaign validation and rebuild decision

Date: 2026-09-08

Branch at inspection: `closure-v2-repairs`

Starting commit: `132f2ac80ab27556f5eabc1957938558dccb0c76`

## Result

The copied CTC data are complete and usable, but the copied node estimates are
not valid inputs to the current artifact builder. No production artifact was
emitted locally.

The four designs contain 147 finalized shards. Three are deliberate duplicate
elastic/equipartition anchors, so the deployable physical grid contains 144
unique `(alpha, theta, aspect_ratio, ensemble_id)` nodes and seven aspect
ratios. All 144 selected raw nodes have schema-2.2 metadata, `_SUCCESS`, both
binary records, and stored `qa_v2.json` records with passing contract status.

The copied estimate JSONs predate the current estimator semantics:

- 147/147 lack `incoming_law_energy`;
- the 36 sentinel estimates also lack `cell_features` and `incoming_law`;
- 0/144 canonical estimates carry the new semantic estimator-contract marker;
- nine low-theta estimates fail the existing closure QA.

The current builder was exercised against a canonical copy of the old
estimates and stopped before table generation with the expected stale-contract
error. This is a fail-closed result, not a failed CTC campaign.

## Why the nodes must be re-estimated

The stability balance is an energy balance. If `z` is the translational share
of the colliding pair's energy, its drift depends on

`<E z_out> - <E z_in>`,

not on the unweighted collision average `<z_out> - <z_in>`. The current gate
therefore requires a two-moment projection of the *energy-weighted* incoming
law. Substituting the older collision-weighted law changes the measure in the
fixed-point equation and is not a metadata-only compatibility issue.

The repeated anchors also need a single policy. The canonical manifest uses
one alpha=1, theta=1 shard per aspect ratio and explicitly passes it to every
fit at that AR. This removes three duplicate physical nodes and prevents an
otherwise constant reference law from changing when the interpolator crosses
from one campaign design to another. The independently measured anchor moments
differ by roughly 2.8e-4 to 5.0e-4 at AR 1.1--1.35: small, but large enough to
avoid deliberately injecting into a near-degenerate exchange problem.

## Existing low-theta warning

The old estimates already identify nine nodes that should be expected to stop
the new build unless the shared-anchor refit changes their verdict:

| alpha | theta | AR | held-out gain (nats/collision) | additional issue |
|---:|---:|---:|---:|---|
| 0.50 | 0.0125 | 1.10 | 0.0431 | lambda3 reached upper bound |
| 0.95 | 0.0125 | 1.35 | 0.0248 | affine memory diagnostic |
| 1.00 | 0.0125 | 1.10 | 0.0229 | -- |
| 1.00 | 0.0125 | 1.20 | 0.0646 | -- |
| 1.00 | 0.0125 | 1.35 | 0.0746 | affine memory diagnostic |
| 1.00 | 0.0250 | 1.20 | 0.0430 | -- |
| 1.00 | 0.0250 | 1.35 | 0.0567 | -- |
| 1.00 | 0.0500 | 1.20 | 0.0243 | -- |
| 1.00 | 0.0500 | 1.35 | 0.0313 | -- |

The model-form threshold is 0.02 nats per accepted collision. These gains are
too large and too systematic across neighboring elastic nodes to dismiss as
Monte Carlo scatter. More CTC samples are not the first remedy.

The likely mathematical issue is that the present scalar Sinkhorn bridge is a
one-memory-parameter, detailed-balance model after the full collision state has
been projected to `z`. Elastic microscopic dynamics are reversible in full
phase space, but the one-step `z` process after marginalizing orientation,
impact geometry, and angular state need not itself be Markovian or satisfy
detailed balance away from equilibrium. Near the sphere limit, exchange is
weak and those hidden variables become long-lived, so the restriction is most
visible at very low theta.

If the current refit reproduces these failures, the next model comparison
should be a stationary but not necessarily detailed-balanced maximum-caliber
bridge. On a fixed `z` grid, fit a positive transition matrix with the measured
equilibrium distribution as an exact invariant, while adding centered memory
bases such as linear and quadratic incoming-partition modes. Compare it to the
current bridge by the same deterministic held-out split. This preserves the
elastic invariant and positivity while testing the missing conditional
curvature directly; it does not weaken or bypass the gate.

## Prepared HPC chain

`hpc/submit_artifact_rebuild.sh` submits one dependency chain:

1. generate the 144-row canonical manifest;
2. refit all 144 nodes with 200 block-bootstrap replicates and 64 propensity
   offsets, using one explicitly shared anchor per AR;
3. replay deep binary validation and require every estimate to satisfy the
   current semantic contract and closure QA;
4. build the artifact only after QA exits successfully.

The expensive stage is the 144-way single-core fit array. A representative
local point fit (zero bootstrap) remained compute-bound after several minutes,
so a production 200-bootstrap grid is not a responsible local job. The raw CTC
trajectories do not need to be regenerated.

The artifact output remains `models/microscopic_closure_v2/`. If any of the
nine structural failures persists, the QA dependency will prevent the artifact
job from starting. The validation report will be written to
`results/closure_estimates/artifact_grid_validation.json`.
