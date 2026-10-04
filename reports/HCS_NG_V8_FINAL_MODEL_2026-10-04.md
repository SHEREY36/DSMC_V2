# HCS-NG protocol v8 with the final model (2026-10-04)

The non-Gaussian HCS campaign of `HCS_NG_PROTOCOL_V8_2026-09-25.md` is rerun with the
released closure (tag `closure-v2-final-v1`, folder `models/microscopic_closure_v2_final_v1`).
Only the model changes.

**Changed**

| | v8 (2026-09-27) | v8, final model |
|---|---|---|
| model variant | `angular_evidence` (contact unit, invariant angular response) | `encounter_final` (one event = one CTC encounter) |
| artifact | `microscopic_closure_v2_angular_evidence/closure_v2.npz` | `microscopic_closure_v2_final_v1/closure_v2.npz` (sha256 `ca99a70e…`) |
| tables | none | `encounter_cross_section.json`, `loss_memory.json`, `angular_memory.json` from the model folder |
| memory per task | 1500M | 1800M (peak RSS 0.98 GiB, was 0.68 GiB) |
| results | `results/hcs_ng_<tag>` | `results/validation/hcs/hcs_ng_<tag>` |

**Unchanged.** Protocol `hcs-ng-v8`, stage designs and gates, coordinates (36, AR 1.1 at
alpha 0.5 still excluded), N = 10^4, tau in [500, 1500] on the collision clock, dt = 0.0025
(0.00125 for the half-step arm), seeds, the two initial-theta branches, the symmetric-midpoint
orientation integrator, the similarity thermostat, the base configuration
`full_domain_baseline_candidate.yaml`, walltimes, QOS and the 256-core throttle. Apart from the
model columns and output paths, the regenerated manifests match the v8 manifests row for row,
column for column (24/120/80/144/360 rows).

The preflight (`hpc/check_hcs_ng_prerequisites.py`) now also checks the model folder for the
`encounter_final` variant: `model_card.json` declares the encounter unit, its hashes match the
artifact and all three tables, and the artifact build passed its HCS stability check.

**Cost.** Measured locally on the most expensive coordinate (alpha 0.5, AR 1.2, theta0 0.025, N = 10^4):

| model | s per collision per particle | physical time per collision per particle |
|---|---|---|
| contact-unit model | 14.6 | 6.20 |
| encounter-unit model | 16.6 | 7.37 |

The encounter clock fires about 1.19 times less often, so each collision covers more simulated time. The measured v8 maxima, 6.2 h (sweep) and 7.2 h
(sentinel half step), become about 7.0 h and 8.1 h, against walltimes of 16 h and 24 h.
