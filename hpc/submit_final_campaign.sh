#!/bin/bash
# The final closure campaign in one command.
#
#   TAG=final_v1_20261003 bash hpc/submit_final_campaign.sh
#     -> models/microscopic_closure_v2_<TAG>/{closure_v2.npz, encounter_cross_section.json,
#                                             angular_memory.json, loss_memory.json}
#
# What is new against alpha_v1_20261001:
#   * every node's exchange kernel is refitted with the post-collision energy
#     weight E_f (estimator contract v4), so each node reproduces the energy
#     each mode gains or loses at any theta;
#   * new theta planes 0.35, 0.5, 0.7 at every AR and alpha (231 CTC nodes),
#     filling the 0.2 -> 1 gap where the near-sphere HCS attractors sit;
#   * the loss table (E[eps|z]) is built into the model folder;
#   * validation: USF (320 runs) and HCS at every corrected-DEM coordinate
#     (130 runs).
#
# Speed: the large arrays run on Negishi's standby QOS (idle cores
# anywhere in the cpu partition, not counted against morri353).  A node fit uses
# 32 cores (parallel bootstrap and propensity): under an hour for a bridge
# node, about two hours for a node that also needs its propensity and the
# bounded-logit repair.  Every
# array skips work that is already finished, so any stage can be resubmitted
# as is -- on standby or, with QOS=normal, on the group allocation.
#
#   CTC theta planes (32 x 20 cores) ---> fits of the new nodes ---+
#   fits of the 396 existing nodes (16 cores each) ---------------+--> deep QA --> precompute --> pack
#   CTC theta planes --> model tables (sigma, angular, loss) ------------------------------------+
#   pack + tables --> production gate (diagnostic) + USF (320) + HCS (130) --> analysis
#
# Restart: RESUME=1 TAG=<same tag> [QOS=normal] bash hpc/submit_final_campaign.sh
#   reuses the manifests and resubmits the chain; finished CTC rows, fits,
#   precompute payloads and validation runs are skipped.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${TAG:?set TAG to a fresh campaign tag}
QOS=${QOS:-standby}
BOOTSTRAP=${BOOTSTRAP:-200}
FIT_CPUS=${FIT_CPUS:-32}
BASE_GRID=${BASE_GRID:-manifests/artifact_grid_alpha_alpha_v1_20261001.csv}
REPLAY_TAG=${REPLAY_TAG:-alpha_v1_20261001}
MODEL=models/microscopic_closure_v2_${TAG}
ESTIMATES=results/closure_estimates/artifact_grid_${TAG}
WORK=results/closure_estimates/artifact_precompute_${TAG}
QA_REPORT=results/closure_estimates/artifact_grid_${TAG}_validation.json
THETA_MANIFEST=manifests/theta_refinement_${TAG}.csv
GRID=manifests/artifact_grid_${TAG}.csv
USF_MANIFEST=manifests/usf_encounter_${TAG}.csv
HCS_MANIFEST=manifests/hcs_encounter_${TAG}.csv
RESUME=${RESUME:-0}
if [[ "$RESUME" != 1 && ( -e "$GRID" || -e "$MODEL/closure_v2.npz" ) ]]; then
  echo "tag $TAG already used; choose a fresh TAG (or resubmit a stage by hand; arrays skip finished work)" >&2
  exit 2
fi
[[ -x HS_CTC_v2/build/SphCyl ]] || { echo "build HS_CTC_v2 first (HS_CTC_v2/build/SphCyl missing)" >&2; exit 2; }
[[ -f "$BASE_GRID" ]] || { echo "missing base grid $BASE_GRID" >&2; exit 2; }
[[ -d results/ctc/nodes/training ]] || { echo "run hpc/organize_results.py --apply first" >&2; exit 2; }
ls -d results/ctc/replays/gate_${REPLAY_TAG}/* >/dev/null 2>&1 \
  || { echo "gate replays results/ctc/replays/gate_${REPLAY_TAG} missing" >&2; exit 2; }
mkdir -p logs manifests "$MODEL"
PY="hpc/python.sh"
export PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src"
$PY hpc/migrate_propensity_cache.py
if [[ "$RESUME" == 1 ]]; then
  for f in "$THETA_MANIFEST" "$GRID" "$USF_MANIFEST" "$HCS_MANIFEST"; do
    [[ -f "$f" ]] || { echo "RESUME=1 needs $f from the first submission" >&2; exit 2; }
  done
else
  $PY hpc/make_theta_refinement_manifest.py --base-grid "$BASE_GRID" --output "$THETA_MANIFEST"
  $PY hpc/prepare_final_campaign.py --base-grid "$BASE_GRID" --theta-manifest "$THETA_MANIFEST" --grid "$GRID"
  $PY DSMC_0D_v2/scripts/make_usf_encounter_manifest.py --tag "$TAG" --ablation-replicates 0 --output "$USF_MANIFEST"
  $PY DSMC_0D_v2/scripts/make_hcs_encounter_manifest.py --tag "$TAG" --arms encounter_memory \
      --dem-reference DSMC_0D_v2/reference/hcs_dem_fresh_v1.csv --replicates 2 --tau-end 100 --output "$HCS_MANIFEST"
fi
GRID_ROWS=$(( $(wc -l < "$GRID") - 1 ))
BASE_ROWS=$(( $(wc -l < "$BASE_GRID") - 1 ))
USF_ROWS=$(( $(wc -l < "$USF_MANIFEST") - 1 ))
HCS_ROWS=$(( $(wc -l < "$HCS_MANIFEST") - 1 ))
id() { printf '%s' "${1%%;*}"; }
Q=(--qos="$QOS")              # large arrays: standby by default
S=(--qos="${SMALL_QOS:-normal}")  # short single jobs start at once on the group allocation

# 1. CTC of the theta planes: 32 x 20 cores, a few minutes per node
CTC=$(id "$(sbatch --parsable "${Q[@]}" --time=03:00:00 --array=0-31 hpc/ctc_stride.slurm "$THETA_MANIFEST")")
# 2. node fits: the existing nodes start now, the new ones after their CTC
FIT_ENV="ALL,CLOSURE_FALLBACK_FORM=conditional_logit_cubic_v3,CLOSURE_BOOTSTRAP=$BOOTSTRAP,CLOSURE_SKIP_CURRENT=1"
FIT_RES=("${Q[@]}" --cpus-per-task="$FIT_CPUS" --mem=$(( 2 * FIT_CPUS ))G --time=04:00:00)
FITS_OLD=$(id "$(sbatch --parsable "${FIT_RES[@]}" --array="0-$((BASE_ROWS - 1))" \
  --export="$FIT_ENV" hpc/closure_fit_array.slurm "$GRID" "$ESTIMATES")")
FITS_NEW=$(id "$(sbatch --parsable "${FIT_RES[@]}" --kill-on-invalid-dep=yes --dependency=afterok:$CTC \
  --array="$BASE_ROWS-$((GRID_ROWS - 1))" --export="$FIT_ENV" hpc/closure_fit_array.slurm "$GRID" "$ESTIMATES")")
# 3. model tables from the training shards (needs the new CTC)
TABLES=$(id "$(sbatch --parsable "${S[@]}" --time=03:00:00 --kill-on-invalid-dep=yes \
  --dependency=afterok:$CTC hpc/build_model_tables.slurm "$MODEL")")
# 4. deep QA of every estimate, precompute (one node per task), pack
QA=$(id "$(sbatch --parsable "${S[@]}" --time=03:00:00 --kill-on-invalid-dep=yes \
  --dependency=afterok:$FITS_OLD:$FITS_NEW hpc/validate_artifact_grid.slurm "$GRID" "$ESTIMATES" "$QA_REPORT")")
PRE=$(id "$(sbatch --parsable "${Q[@]}" --time=03:00:00 --kill-on-invalid-dep=yes --dependency=afterok:$QA \
  --array="0-$((GRID_ROWS - 1))" --export=ALL,PRECOMPUTE_SKIP_EXISTING=1 \
  hpc/artifact_precompute_stride.slurm "$GRID" "$ESTIMATES" "$WORK" "$GRID_ROWS")")
PACK=$(id "$(sbatch --parsable "${S[@]}" --time=04:00:00 --kill-on-invalid-dep=yes --dependency=afterok:$PRE \
  --export=ALL,ARTIFACT_EVENT_UNIT=encounter hpc/aggregate.slurm "$GRID" "$ESTIMATES" "$MODEL" "$WORK")")
# 5. diagnostic gate on the existing replays, validation, analysis
TABLE_ENV="ALL,ENCOUNTER_TABLE=$MODEL/encounter_cross_section.json,ANGULAR_MEMORY=$MODEL/angular_memory.json,LOSS_MEMORY=$MODEL/loss_memory.json"
GATE=$(id "$(sbatch --parsable "${S[@]}" --time=02:00:00 --kill-on-invalid-dep=yes \
  --dependency=afterok:$PACK:$TABLES --export=ALL,REPLAY_TAG=$REPLAY_TAG \
  hpc/production_gate.slurm "$TAG" "$MODEL")")
USF=$(id "$(sbatch --parsable "${Q[@]}" --time=03:30:00 --kill-on-invalid-dep=yes --dependency=afterok:$PACK:$TABLES \
  --array="0-$((USF_ROWS - 1))" --export="$TABLE_ENV,USF_ENC_STRIDE=$USF_ROWS" \
  hpc/usf_encounter_array.slurm "$USF_MANIFEST" "$MODEL/closure_v2.npz")")
HCS=$(id "$(sbatch --parsable "${Q[@]}" --time=04:00:00 --kill-on-invalid-dep=yes --dependency=afterok:$PACK:$TABLES \
  --array="0-$((HCS_ROWS - 1))" --export="$TABLE_ENV" \
  hpc/hcs_encounter_array.slurm "$HCS_MANIFEST" "$MODEL/closure_v2.npz")")
ANALYSIS=$(id "$(sbatch --parsable "${S[@]}" --kill-on-invalid-dep=yes --dependency=afterany:$USF:$HCS \
  --export="$TABLE_ENV" hpc/analyze_usf_encounter.slurm "$TAG")")
cat <<MSG
final campaign $TAG on QOS $QOS   model folder $MODEL
  CTC theta planes  $CTC        ($(( $(wc -l < "$THETA_MANIFEST") - 1 )) nodes)
  fits, existing    $FITS_OLD   ($BASE_ROWS nodes, $FIT_CPUS cores each, $BOOTSTRAP bootstraps)
  fits, new theta   $FITS_NEW   ($((GRID_ROWS - BASE_ROWS)) nodes)
  model tables      $TABLES
  deep QA           $QA         -> $QA_REPORT
  precompute        $PRE        ($GRID_ROWS nodes)
  pack              $PACK
  production gate   $GATE       -> results/validation/production_gate/${TAG}/gate.{csv,json}
  USF validation    $USF        ($USF_ROWS runs)
  HCS validation    $HCS        ($HCS_ROWS runs)
  analysis          $ANALYSIS   -> results/validation/usf/usf_encounter_${TAG}/usf_vs_dem.*
                                   results/validation/hcs/hcs_encounter_${TAG}/hcs_vs_dem.csv
MSG
