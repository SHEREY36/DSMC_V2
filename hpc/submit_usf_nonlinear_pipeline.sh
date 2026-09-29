#!/bin/bash
# Submit one end-to-end, fail-closed nonlinear USF closure campaign on Negishi.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
# v1 replay reservoirs were invalidated by a decimal-filename collision. Keep
# the recovery campaign on a fresh namespace even when TAG is not supplied.
TAG=${TAG:-usf_nonlinear_v2_20260928}
BASE_ARTIFACT=${BASE_ARTIFACT:-models/microscopic_closure_v2_angular_evidence/closure_v2.npz}
HARVEST_MANIFEST=manifests/${TAG}_harvest.csv
HARVEST_RESULTS=results/${TAG}/harvest
CTC_MANIFEST=manifests/${TAG}_ctc.csv
CTC_RESULTS=outputs/${TAG}_ctc
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

# A non-submitting contract test may redirect every newly generated path to
# /tmp and replace sbatch with deterministic placeholder IDs. Production uses
# the repository paths and real scheduler unchanged.
if [[ -n "${CAMPAIGN_WORK_ROOT:-}" ]]; then
  WORK_ROOT=${CAMPAIGN_WORK_ROOT%/}
  HARVEST_MANIFEST=$WORK_ROOT/manifests/${TAG}_harvest.csv
  HARVEST_RESULTS=$WORK_ROOT/results/${TAG}/harvest
  CTC_MANIFEST=$WORK_ROOT/manifests/${TAG}_ctc.csv
  CTC_RESULTS=$WORK_ROOT/outputs/${TAG}_ctc
  DIRECT_ESTIMATES=$WORK_ROOT/results/closure_estimates/${TAG}_direct
  LOCAL_COEFFICIENTS=$WORK_ROOT/results/closure_estimates/${TAG}_quadratic_local.json
  FINAL_COEFFICIENTS=$WORK_ROOT/results/closure_estimates/${TAG}_quadratic_direct.json
  DIRECT_REPORT=$WORK_ROOT/results/closure_estimates/${TAG}_direct_report.json
  PRECOMPUTED=$WORK_ROOT/results/closure_estimates/${TAG}_precompute
  CANDIDATE_DIR=$WORK_ROOT/models/microscopic_closure_v2_${TAG}
  CANDIDATE=$CANDIDATE_DIR/closure_v2.npz
  HCS_MANIFEST=$WORK_ROOT/manifests/${TAG}_hcs_validation.csv
  HCS_RESULTS=$WORK_ROOT/results/${TAG}/hcs_validation
  HCS_SUMMARY=$HCS_RESULTS/summary.json
  USF_MANIFEST=$WORK_ROOT/manifests/${TAG}_usf_validation.csv
  USF_RESULTS=$WORK_ROOT/results/${TAG}/usf_validation
  USF_SUMMARY=$USF_RESULTS/summary.json
  PROMOTION=$WORK_ROOT/results/${TAG}/promotion.json
fi

submit_job() {
  if [[ "${PIPELINE_DRY_RUN:-false}" == "true" ]]; then
    printf '%s\n' 990000
  else
    sbatch "$@"
  fi
}

mkdir -p logs manifests "results/$TAG" results/closure_estimates
for path in "$BASE_ARTIFACT" "$GRID" DSMC_0D_v2/reference/usf_dem_fresh_v1.csv; do
  [[ -f "$path" ]] || { echo "required input missing: $path" >&2; exit 2; }
done
for directory in "$BASELINES" "${EXCITATION_DIRS[@]}"; do
  [[ -d "$directory" ]] || { echo "required estimate directory missing: $directory" >&2; exit 2; }
done
[[ $(( $(wc -l < "$GRID") - 1 )) -eq 144 ]] || {
  echo "canonical artifact grid must have 144 rows" >&2; exit 2; }
[[ ! -e "$CANDIDATE_DIR" && ! -e "$HARVEST_RESULTS" \
   && ! -e "$CTC_RESULTS" && ! -e "$DIRECT_ESTIMATES" ]] || {
  echo "tag $TAG already has outputs; choose a fresh TAG or use the documented retry tools" >&2
  exit 2
}

hpc/python.sh hpc/verify_python_environment.py
hpc/python.sh -m compileall -q contracts/python Coll_Models_v2/src DSMC_0D_v2/src \
  hpc DSMC_0D_v2/scripts Coll_Models_v2/scripts
hpc/python.sh hpc/make_usf_response_manifest.py --tag "$TAG" \
  --output "$HARVEST_MANIFEST" --results "$HARVEST_RESULTS" \
  --artifact "$BASE_ARTIFACT" --particles "${HARVEST_PARTICLES:-50000}" \
  --reservoir-capacity "${REPLAY_CAPACITY:-200000}"
[[ $(( $(wc -l < "$HARVEST_MANIFEST") - 1 )) -eq 42 ]] || {
  echo "harvest manifest must have 42 rows" >&2; exit 2; }

# Build CTC while the independent DSMC trajectories are harvesting states.
BUILD_RAW=$(submit_job --parsable hpc/build_usf_response_ctc.slurm)
BUILD_JOB=${BUILD_RAW%%;*}
HARVEST_RAW=$(submit_job --parsable --array=0-41%42 \
  hpc/usf_response_harvest_array.slurm "$HARVEST_MANIFEST")
HARVEST_JOB=${HARVEST_RAW%%;*}
PREP_CTC_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HARVEST_JOB" hpc/prepare_usf_response_ctc.slurm \
  "$HARVEST_MANIFEST" "$CTC_MANIFEST" "$CTC_RESULTS")
PREP_CTC_JOB=${PREP_CTC_RAW%%;*}
CTC_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$BUILD_JOB:$PREP_CTC_JOB" --array=0-167%64 \
  hpc/usf_response_ctc_array.slurm "$CTC_MANIFEST")
CTC_JOB=${CTC_RAW%%;*}
ESTIMATE_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$CTC_JOB" --array=0-167%85 \
  hpc/usf_response_estimate_array.slurm "$CTC_MANIFEST" "$DIRECT_ESTIMATES")
ESTIMATE_JOB=${ESTIMATE_RAW%%;*}
ANALYZE_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$ESTIMATE_JOB" hpc/analyze_usf_response.slurm \
  "$LOCAL_COEFFICIENTS" "$DIRECT_ESTIMATES" "$FINAL_COEFFICIENTS" \
  "$DIRECT_REPORT" "$BASE_ARTIFACT")
ANALYZE_JOB=${ANALYZE_RAW%%;*}

# Rebuild all energy tables against the enlarged, quadratic response bounds.
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
VALIDATION_PREP_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PACK_JOB" hpc/prepare_usf_response_validation.slurm \
  "$CANDIDATE" "$HCS_MANIFEST" "$HCS_RESULTS" "$USF_MANIFEST" "$USF_RESULTS")
VALIDATION_PREP_JOB=${VALIDATION_PREP_RAW%%;*}

# At most 112 HCS tasks and 144 of the 276 USF tasks coexist at 256 cores.
HCS_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$VALIDATION_PREP_JOB" --array=0-231%112 \
  hpc/hcs_validation_array.slurm "$HCS_MANIFEST" "$CANDIDATE" true true true)
HCS_JOB=${HCS_RAW%%;*}
USF_RAW=$(submit_job --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$VALIDATION_PREP_JOB" --array=0-275%144 \
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
tag=$TAG
ctc_build=$BUILD_JOB
DSMC_harvest=$HARVEST_JOB (42 x 1 core; N=${HARVEST_PARTICLES:-50000})
ctc_manifest=$PREP_CTC_JOB
exact_CTC=$CTC_JOB (168 tasks; 64 x 4 cores = 256)
direct_estimation=$ESTIMATE_JOB (85 x 3 cores = 255)
direct_response_gate=$ANALYZE_JOB
candidate_precompute=$PRE_JOB (128 x 2 cores = 256)
candidate_pack=$PACK_JOB
validation_prepare=$VALIDATION_PREP_JOB
HCS_validation=$HCS_JOB (232 tasks, at most 112 cores)
USF_validation=$USF_JOB (276 tasks, at most 144 cores alongside HCS)
HCS_QA=$HCS_QA_JOB USF_QA=$USF_QA_JOB
promotion_gate=$PROMOTE_JOB
EOF
