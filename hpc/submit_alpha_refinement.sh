#!/bin/bash
# One-command alpha refinement of the encounter-unit closure.
#
#   TAG=alpha_v1_20261001 bash hpc/submit_alpha_refinement.sh   (model -> models/microscopic_closure_v2_alpha_v1_20261001)
#
# Chain (Slurm dependencies):
#   CTC new alpha planes ─┬─> fit new nodes ─> deep QA ─> precompute ─> pack ─┐
#                         └─> model tables (sigma, <k>, angular memory) ──────┤
#   CTC gate replays (USF pairs at node and between-node alphas) ─────────────┤
#                                   production gate <─────────────────────────┤
#                                   USF + HCS validation <────────────────────┘ -> analysis
# The model folder models/microscopic_closure_v2_<TAG>/ holds the artifact
# and its tables (the configuration pattern is DSMC_0D_v2/config/encounter_unit_model_final.yaml).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${TAG:?set TAG to a fresh campaign tag}
ALPHAS=${ALPHAS:-0.55,0.6,0.65,0.7,0.75,0.85,0.9}
# Bootstrap resamples per new node: 200 matches the existing 144 fits (about 7 h
# per node on one core); 100 halves the fit stage.
BOOTSTRAP=${BOOTSTRAP:-200}
MODEL=models/microscopic_closure_v2_${TAG}
ESTIMATES=results/closure_estimates/artifact_grid_alpha_${TAG}
WORK=results/closure_estimates/artifact_precompute_alpha_${TAG}
QA_REPORT=results/closure_estimates/artifact_grid_alpha_${TAG}_validation.json
CTC_MANIFEST=manifests/alpha_refinement_${TAG}.csv
GRID=manifests/artifact_grid_alpha_${TAG}.csv
FIT=manifests/artifact_grid_alpha_${TAG}_fit.csv
GATE_MANIFEST=manifests/production_gate_${TAG}.csv
USF_MANIFEST=manifests/usf_encounter_${TAG}.csv
HCS_MANIFEST=manifests/hcs_encounter_${TAG}.csv
if [[ -e "$CTC_MANIFEST" || -e "$MODEL/closure_v2.npz" ]]; then
  echo "tag $TAG already used; choose a fresh TAG (arrays skip finished rows if resubmitted by hand)" >&2
  exit 2
fi
[[ -x HS_CTC_v2/build/SphCyl ]] || { echo "build HS_CTC_v2 first (HS_CTC_v2/build/SphCyl missing)" >&2; exit 2; }
for f in manifests/artifact_grid.csv manifests/artifact_grid_repairs.csv manifests/closure_sentinel.csv \
         manifests/ar_extension.csv manifests/ar_near_sphere.csv manifests/ar_low_theta.csv; do
  [[ -f "$f" ]] || { echo "missing canonical grid manifest $f" >&2; exit 2; }
done
ls results/dsmc_harvest/usf_nonlinear_v2_20260928/AR_3.000_alpha_0.800_cold_replay_window_02.bin >/dev/null \
  || { echo "USF harvest reservoirs (usf_nonlinear_v2_20260928) are needed for the gate" >&2; exit 2; }
[[ $(ls results/closure_estimates/artifact_grid/*.json | wc -l) -ge 144 ]] \
  || { echo "results/closure_estimates/artifact_grid must hold the 144 current node estimates" >&2; exit 2; }
mkdir -p logs manifests "$MODEL"
PY="hpc/python.sh"
export PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src"
$PY hpc/make_alpha_refinement_manifest.py --alphas "$ALPHAS" --output "$CTC_MANIFEST"
$PY hpc/prepare_alpha_refinement.py --refinement-manifest "$CTC_MANIFEST" \
    --estimates "$ESTIMATES" --grid "$GRID" --fit "$FIT"
$PY hpc/make_gate_replay_manifest.py --tag "$TAG" --output "$GATE_MANIFEST"
$PY DSMC_0D_v2/scripts/make_usf_encounter_manifest.py --tag "$TAG" --ablation-replicates 0 --output "$USF_MANIFEST"
$PY DSMC_0D_v2/scripts/make_hcs_encounter_manifest.py --tag "$TAG" --arms encounter_memory --output "$HCS_MANIFEST"
GRID_ROWS=$(( $(wc -l < "$GRID") - 1 ))
FIT_ROWS=$(( $(wc -l < "$FIT") - 1 ))
USF_ROWS=$(( $(wc -l < "$USF_MANIFEST") - 1 ))
HCS_ROWS=$(( $(wc -l < "$HCS_MANIFEST") - 1 ))
id() { printf '%s' "${1%%;*}"; }

# 1. CTC: 8 x 20-core stride workers (160 cores) and the gate replays alongside
#    them on 12 x 8 cores (96 cores): 256 cores in total.
CTC=$(id "$(sbatch --parsable --array=0-7 hpc/ctc_stride.slurm "$CTC_MANIFEST")")
GATE_CTC=$(id "$(sbatch --parsable --array=0-11 hpc/gate_replay_array.slurm "$GATE_MANIFEST")")
# 2. fit the new nodes (bridge form; bounded-logit repair when a node fails QA)
FITS=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$CTC \
  --array="0-$((FIT_ROWS - 1))%256" \
  --export=ALL,CLOSURE_FALLBACK_FORM=conditional_logit_cubic_v3,CLOSURE_BOOTSTRAP=$BOOTSTRAP \
  hpc/closure_fit_array.slurm "$FIT" "$ESTIMATES")")
TABLES=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$CTC \
  hpc/build_model_tables.slurm "$MODEL")")
QA=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$FITS \
  hpc/validate_artifact_grid.slurm "$GRID" "$ESTIMATES" "$QA_REPORT")")
PRE_TASKS=$GRID_ROWS
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
(( PRE_TASKS > ${MAX_ARRAY:-1000} )) && PRE_TASKS=${MAX_ARRAY:-1000}
PRE=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$QA \
  --array="0-$((PRE_TASKS - 1))%128" \
  hpc/artifact_precompute_stride.slurm "$GRID" "$ESTIMATES" "$WORK" "$GRID_ROWS")")
PACK=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PRE \
  --export=ALL,ARTIFACT_EVENT_UNIT=encounter \
  hpc/aggregate.slurm "$GRID" "$ESTIMATES" "$MODEL" "$WORK")")
# 3. production gate and validation on the packed model
GATE=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PACK:$TABLES,afterany:$GATE_CTC \
  hpc/production_gate.slurm "$TAG" "$MODEL")")
TABLE_ENV="ALL,ENCOUNTER_TABLE=$MODEL/encounter_cross_section.json,ANGULAR_MEMORY=$MODEL/angular_memory.json"
# the loss follows the incoming split whenever the model folder carries the table
[[ -f "$MODEL/loss_memory.json" || "$0" == *submit_alpha_refinement.sh ]] && TABLE_ENV="$TABLE_ENV,LOSS_MEMORY=$MODEL/loss_memory.json"
USF_WORKERS=208; (( USF_WORKERS > USF_ROWS )) && USF_WORKERS=$USF_ROWS
USF=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PACK:$TABLES \
  --array="0-$((USF_WORKERS - 1))" --export="$TABLE_ENV,USF_ENC_STRIDE=$USF_WORKERS" \
  hpc/usf_encounter_array.slurm "$USF_MANIFEST" "$MODEL/closure_v2.npz")")
HCS=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PACK:$TABLES \
  --array="0-$((HCS_ROWS - 1))%48" --export="$TABLE_ENV" \
  hpc/hcs_encounter_array.slurm "$HCS_MANIFEST" "$MODEL/closure_v2.npz")")
ANALYSIS=$(id "$(sbatch --parsable --dependency=afterany:$USF:$HCS --export="$TABLE_ENV" \
  hpc/analyze_usf_encounter.slurm "$TAG")")
cat <<MSG
tag $TAG   model folder $MODEL
  ctc planes      $CTC        ($(( $(wc -l < "$CTC_MANIFEST") - 1 )) nodes at alpha $ALPHAS)
  gate replays    $GATE_CTC   ($(( $(wc -l < "$GATE_MANIFEST") - 1 )) CTC replays of USF pairs)
  node fits       $FITS       ($FIT_ROWS new nodes; $((GRID_ROWS - FIT_ROWS)) reused)
  model tables    $TABLES
  deep QA         $QA
  precompute      $PRE        ($GRID_ROWS nodes)
  pack            $PACK
  production gate $GATE       -> results/validation/production_gate/${TAG}/gate.{csv,json}
  USF validation  $USF        ($USF_ROWS runs)  HCS validation $HCS ($HCS_ROWS runs)
  analysis        $ANALYSIS   -> results/validation/usf/usf_encounter_${TAG}/usf_vs_dem.*  results/validation/hcs/hcs_encounter_${TAG}/hcs_vs_dem.csv
MSG
