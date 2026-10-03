#!/bin/bash
# Resume the v2 nonlinear campaign after its completed harvest/CTC/estimator
# stages. No collision data or point estimates are recomputed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
TAG=${TAG:-usf_nonlinear_v2_20260928}
BASE_ARTIFACT=${BASE_ARTIFACT:-models/microscopic_closure_v2_angular_evidence/closure_v2.npz}
CTC_RESULTS=outputs/${TAG}_ctc/replay
DIRECT_ESTIMATES=results/closure_estimates/${TAG}_direct
LOCAL_COEFFICIENTS=results/closure_estimates/${TAG}_quadratic_local.json
FINAL_COEFFICIENTS=results/closure_estimates/${TAG}_quadratic_direct.json
DIRECT_REPORT=results/closure_estimates/${TAG}_direct_report.json
PRECOMPUTED=results/closure_estimates/${TAG}_precompute
CANDIDATE_DIR=models/microscopic_closure_v2_${TAG}
CANDIDATE=$CANDIDATE_DIR/closure_v2.npz
HCS_MANIFEST=manifests/${TAG}_hcs_validation.csv
HCS_RESULTS=results/${TAG}/hcs_validation
HCS_SUMMARY=$HCS_RESULTS/summary.json
USF_MANIFEST=manifests/${TAG}_usf_validation.csv
USF_RESULTS=results/${TAG}/usf_validation
USF_SUMMARY=$USF_RESULTS/summary.json
PROMOTION=results/${TAG}/promotion.json
GRID=manifests/artifact_grid.csv
BASELINES=results/closure_estimates/artifact_grid
EXCITATION_DIRS=(
  results/closure_estimates/excitation_correction-grid
  results/closure_estimates/excitation_usf_extension_v1
  results/closure_estimates/excitation_full_candidate_v1_support_refinement
  results/closure_estimates/excitation_full_candidate_v1_missing
)

submit_job() {
  if [[ "${PIPELINE_DRY_RUN:-false}" == "true" ]]; then
    printf '%s\n' 990001
  else
    sbatch "$@"
  fi
}

mkdir -p logs
for path in "$BASE_ARTIFACT" "$GRID" "$LOCAL_COEFFICIENTS"; do
  [[ -f "$path" ]] || { echo "required completed input missing: $path" >&2; exit 2; }
done
[[ -d "$DIRECT_ESTIMATES" && -d "$CTC_RESULTS" ]] || {
  echo "completed direct estimates or CTC replay directory is missing" >&2; exit 2; }
[[ $(find "$DIRECT_ESTIMATES" -maxdepth 1 -type f -name 'replay_*.json' | wc -l) -eq 168 ]] || {
  echo "resume requires exactly 168 direct estimates" >&2; exit 2; }
[[ $(find "$CTC_RESULTS" -mindepth 2 -maxdepth 2 -type f -name _SUCCESS | wc -l) -eq 168 ]] || {
  echo "resume requires exactly 168 finalized CTC replays" >&2; exit 2; }
[[ ! -e "$FINAL_COEFFICIENTS" && ! -e "$DIRECT_REPORT" \
   && ! -e "$PRECOMPUTED" && ! -e "$CANDIDATE_DIR" \
   && ! -e "$HCS_RESULTS" && ! -e "$USF_RESULTS" \
   && ! -e "$PROMOTION" ]] || {
  echo "downstream $TAG outputs already exist; inspect them before any retry" >&2
  exit 2
}

hpc/python.sh hpc/verify_python_environment.py
hpc/python.sh -m compileall -q contracts/python Coll_Models_v2/src \
  DSMC_0D_v2/src hpc DSMC_0D_v2/scripts Coll_Models_v2/scripts

ANALYZE_RAW=$(submit_job --parsable hpc/analyze_usf_response.slurm \
  "$LOCAL_COEFFICIENTS" "$DIRECT_ESTIMATES" "$FINAL_COEFFICIENTS" \
  "$DIRECT_REPORT" "$BASE_ARTIFACT")
ANALYZE_JOB=${ANALYZE_RAW%%;*}
PRE_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$ANALYZE_JOB" --array=0-127%128 \
  --export="ALL,COEFFICIENT_ROWS=$FINAL_COEFFICIENTS,COEFFICIENT_RELEASE_POLICY=quadratic-evidence-v1" \
  hpc/artifact_precompute_stride.slurm "$GRID" "$BASELINES" "$PRECOMPUTED" 144 \
  "${EXCITATION_DIRS[@]}")
PRE_JOB=${PRE_RAW%%;*}
PACK_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PRE_JOB" \
  --export="ALL,COEFFICIENT_ROWS=$FINAL_COEFFICIENTS,COEFFICIENT_RELEASE_POLICY=quadratic-evidence-v1" \
  hpc/aggregate.slurm "$GRID" "$BASELINES" "$CANDIDATE_DIR" "$PRECOMPUTED" \
  "${EXCITATION_DIRS[@]}")
PACK_JOB=${PACK_RAW%%;*}
PREP_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PACK_JOB" hpc/prepare_usf_response_validation.slurm \
  "$CANDIDATE" "$HCS_MANIFEST" "$HCS_RESULTS" "$USF_MANIFEST" "$USF_RESULTS")
PREP_JOB=${PREP_RAW%%;*}
HCS_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PREP_JOB" --array=0-231%112 \
  hpc/hcs_validation_array.slurm "$HCS_MANIFEST" "$CANDIDATE" true true true)
HCS_JOB=${HCS_RAW%%;*}
USF_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PREP_JOB" --array=0-275%144 \
  --export=ALL,USF_STUDY_STRIDE=276 \
  hpc/usf_study_array.slurm "$USF_MANIFEST" "$CANDIDATE")
USF_JOB=${USF_RAW%%;*}
HCS_QA_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HCS_JOB" hpc/plot_hcs_validation.slurm \
  "$HCS_MANIFEST" "$HCS_RESULTS/hcs_theta_attraction.png" "$HCS_SUMMARY")
HCS_QA_JOB=${HCS_QA_RAW%%;*}
USF_QA_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$USF_JOB" hpc/analyze_usf_study.slurm \
  "$USF_MANIFEST" "$USF_SUMMARY" "$USF_RESULTS/usf_response_validation.png")
USF_QA_JOB=${USF_QA_RAW%%;*}
PROMOTE_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HCS_QA_JOB:$USF_QA_JOB" \
  hpc/finalize_usf_response_candidate.slurm "$CANDIDATE" "$DIRECT_REPORT" \
  "$HCS_SUMMARY" "$USF_SUMMARY" "$PROMOTION")
PROMOTE_JOB=${PROMOTE_RAW%%;*}

cat <<EOF
tag=$TAG (reusing 42 harvests, 168 CTC replays, and 168 estimates)
direct_response_reanalysis=$ANALYZE_JOB
candidate_precompute=$PRE_JOB (128 x 2 cores = 256)
candidate_pack=$PACK_JOB
validation_prepare=$PREP_JOB
HCS_validation=$HCS_JOB (232 tasks, at most 112 cores)
USF_validation=$USF_JOB (276 tasks, at most 144 cores alongside HCS)
HCS_QA=$HCS_QA_JOB USF_QA=$USF_QA_JOB
promotion_gate=$PROMOTE_JOB
EOF
