#!/bin/bash
# Resume the v2 nonlinear campaign after the response gate and all 144
# artifact-node precomputations completed. No DSMC, CTC, response estimation,
# coefficient fitting, or artifact geometry is recomputed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
TAG=${TAG:-usf_nonlinear_v2_20260928}
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
for path in "$FINAL_COEFFICIENTS" "$DIRECT_REPORT" "$GRID"; do
  [[ -f "$path" ]] || { echo "required completed input missing: $path" >&2; exit 2; }
done
[[ -d "$BASELINES" && -d "$PRECOMPUTED" ]] || {
  echo "baseline estimates or completed precompute directory is missing" >&2; exit 2; }

# Count and name every expected payload. A simple count would not detect a
# duplicated/misnumbered copy. The packer independently checks coordinates,
# estimate digests, correction digest, schema, and numerical table shapes.
[[ $(find "$PRECOMPUTED" -maxdepth 1 -type f -name 'node_*.npz' | wc -l) -eq 144 ]] || {
  echo "resume requires exactly 144 artifact precompute payloads" >&2; exit 2; }
for (( index=0; index<144; index++ )); do
  printf -v payload '%s/node_%04d.npz' "$PRECOMPUTED" "$index"
  [[ -s "$payload" ]] || { echo "missing precompute payload: $payload" >&2; exit 2; }
done

hpc/python.sh -c '
import json, sys
report = json.load(open(sys.argv[1]))
coefficients = json.load(open(sys.argv[2]))
if not report.get("candidate_build_ready", False):
    raise SystemExit("direct-response report is not candidate-build ready")
if report.get("n_direct_estimates") != 168:
    raise SystemExit("direct-response report does not cover 168 estimates")
if coefficients.get("release_policy") != "quadratic-evidence-v1":
    raise SystemExit("unexpected frozen coefficient release policy")
if coefficients.get("n_coefficient_nodes") != 144:
    raise SystemExit("frozen coefficient surface does not cover 144 nodes")
' "$DIRECT_REPORT" "$FINAL_COEFFICIENTS"

[[ ! -e "$CANDIDATE" ]] || {
  echo "candidate artifact already exists; inspect it before any retry" >&2; exit 2; }
if [[ -d "$CANDIDATE_DIR" ]] \
    && [[ -n $(find "$CANDIDATE_DIR" -mindepth 1 -print -quit) ]]; then
  echo "partial candidate directory is not empty; inspect it before any retry" >&2
  exit 2
fi
for path in "$HCS_MANIFEST" "$USF_MANIFEST" "$HCS_RESULTS" \
            "$USF_RESULTS" "$PROMOTION"; do
  [[ ! -e "$path" ]] || {
    echo "downstream output already exists: $path; inspect before retry" >&2; exit 2; }
done

hpc/python.sh hpc/verify_python_environment.py
hpc/python.sh -m compileall -q contracts/python Coll_Models_v2/src \
  DSMC_0D_v2/src hpc DSMC_0D_v2/scripts Coll_Models_v2/scripts

PACK_RAW=$(submit_job --parsable \
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
tag=$TAG (reusing response fit and all 144 artifact precomputations)
candidate_pack=$PACK_JOB
validation_prepare=$PREP_JOB
HCS_validation=$HCS_JOB (232 tasks, at most 112 cores)
USF_validation=$USF_JOB (276 tasks, at most 144 cores alongside HCS)
HCS_QA=$HCS_QA_JOB USF_QA=$USF_QA_JOB
promotion_gate=$PROMOTE_JOB
EOF
