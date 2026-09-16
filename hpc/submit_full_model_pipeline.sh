#!/bin/bash
# One-shot completion of the correction surface, artifact, HCS/performance,
# fresh-CTC, USF, and non-Gaussian domain-pilot validation chain.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${TAG:-full_candidate_v1}
RESUME_FROM_MISSING=${RESUME_FROM_MISSING:-false}
GRID=${GRID:-manifests/artifact_grid.csv}
BASELINES=${BASELINES:-results/closure_estimates/artifact_grid}
OLD_MANIFEST=${OLD_MANIFEST:-manifests/excitation_correction-grid.csv}
OLD_RESULTS=${OLD_RESULTS:-results/closure_estimates/excitation_correction-grid}
EXT_MANIFEST=${EXT_MANIFEST:-manifests/excitation_usf_extension_v1.csv}
EXT_RESULTS=${EXT_RESULTS:-results/closure_estimates/excitation_usf_extension_v1}
MISSING_MANIFEST="manifests/excitation_${TAG}_missing.csv"
MISSING_RESULTS="results/closure_estimates/excitation_${TAG}_missing"
MISSING_RETRY_MANIFEST="manifests/excitation_${TAG}_missing_retry.csv"
REFINE_MANIFEST="manifests/excitation_${TAG}_support_refinement.csv"
REFINE_RESULTS="results/closure_estimates/excitation_${TAG}_support_refinement"
COMBINED_MANIFEST="manifests/excitation_${TAG}_combined.csv"
COMBINED_SUMMARY="results/closure_estimates/excitation_${TAG}_combined_summary.json"
COMBINED_OFFLINE="results/closure_estimates/excitation_${TAG}_combined_offline_validation.json"
COEFFICIENT_ROWS="results/closure_estimates/excitation_${TAG}_coefficient_surface.json"
PRECOMPUTED="results/closure_estimates/artifact_precompute_${TAG}"
ARTIFACT_DIR="models/microscopic_closure_v2_${TAG}"
ARTIFACT="$ARTIFACT_DIR/closure_v2.npz"
HCS_REF_MANIFEST="manifests/hcs_${TAG}_reference.csv"
HCS_OPT_MANIFEST="manifests/hcs_${TAG}_optimized.csv"
HCS_FULL_RESULTS="results/hcs_${TAG}_full_domain"
HCS_FULL_SUMMARY="$HCS_FULL_RESULTS/summary.json"
CADENCE_RESULT="results/hcs_${TAG}_cadence_validation.json"
HOLDOUT_MANIFEST=${HOLDOUT_MANIFEST:-manifests/excitation_independent_holdout_v1.csv}
HOLDOUT_BASELINE=${HOLDOUT_BASELINE:-results/closure_estimates/independent_ctc_holdout_v1_baseline/alpha_0.950_theta_1.000_AR_2.000_ensemble_000.json}
HOLDOUT_VALIDATION="results/closure_estimates/independent_ctc_holdout_${TAG}_validation.json"
USF_PILOT_MANIFEST="manifests/usf_${TAG}_pilot.csv"
USF_PILOT_RESULTS="results/usf_${TAG}_pilot"
USF_FULL_MANIFEST="manifests/usf_${TAG}_full.csv"
USF_FULL_RESULTS="results/usf_${TAG}_full"
NG_MANIFEST="manifests/hcs_ng_${TAG}_domain_pilot.csv"
NG_RESULTS="results/hcs_ng_${TAG}_domain_pilot"

mkdir -p logs results/closure_estimates
# Fail before submitting any array if the selected immutable environment cannot
# load and execute the SciPy components used by fit and artifact workers.
hpc/python.sh hpc/verify_python_environment.py
FRESH_TARGETS=("$PRECOMPUTED" "$ARTIFACT_DIR" "results/hcs_${TAG}_reference" \
  "$HCS_FULL_RESULTS" "$USF_PILOT_RESULTS" "$USF_FULL_RESULTS" "$NG_RESULTS")
if [[ "$RESUME_FROM_MISSING" != "true" ]]; then
  FRESH_TARGETS=("$MISSING_RESULTS" "$REFINE_RESULTS" "${FRESH_TARGETS[@]}")
fi
for target in "${FRESH_TARGETS[@]}"; do
  if [[ -e "$target" ]] && find "$target" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "refusing to mix this run with existing outputs in $target" >&2
    exit 2
  fi
done
for target in "$COMBINED_SUMMARY" "$COMBINED_OFFLINE" \
  "$COEFFICIENT_ROWS" \
  "$CADENCE_RESULT" "$HOLDOUT_VALIDATION"; do
  [[ ! -e "$target" ]] || { echo "refusing to overwrite $target" >&2; exit 2; }
done

# Validate all reusable evidence before purchasing another CPU-hour.
hpc/python.sh hpc/require_excitation_pass.py \
  results/closure_estimates/excitation_correction-grid_summary.json \
  --expected-mode correction-grid
hpc/python.sh hpc/require_excitation_pass.py \
  results/closure_estimates/excitation_usf_extension_v1_summary.json \
  --expected-mode usf-extension
hpc/python.sh -c \
  'import json,sys; assert json.load(open(sys.argv[1]))["validation_pass"]' \
  results/closure_estimates/independent_ctc_holdout_v1_validation.json
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/validate_artifact_inputs.py --manifest "$GRID" \
  --estimates "$BASELINES" \
  --output "results/closure_estimates/${TAG}_baseline_validation.json" \
  --require-current-estimates --require-pass

# Generate only the 108 physical correction nodes not already represented by
# the two completed 18-node campaigns.  Add an intermediate ±0.35 validation
# amplitude everywhere so support can be released without guessing between
# the fitted ±0.25 and outer held-out ±0.50 points.
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/make_excitation_manifest.py --mode production-grid \
  --grid "$GRID" --estimates "$BASELINES" --output "$MISSING_MANIFEST" \
  --results "$MISSING_RESULTS" --exclude-manifest "$OLD_MANIFEST" \
  --exclude-manifest "$EXT_MANIFEST" \
  --amplitudes -0.50 -0.35 -0.25 0.25 0.35 0.50
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/make_excitation_manifest.py --mode support-refinement \
  --grid "$GRID" --estimates "$BASELINES" --output "$REFINE_MANIFEST" \
  --results "$REFINE_RESULTS" --include-manifest "$OLD_MANIFEST" \
  --include-manifest "$EXT_MANIFEST"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/combine_excitation_manifests.py --output "$COMBINED_MANIFEST" \
  "$OLD_MANIFEST" "$EXT_MANIFEST" "$REFINE_MANIFEST" "$MISSING_MANIFEST"
MISSING_ROWS=$(( $(wc -l < "$MISSING_MANIFEST") - 1 ))
REFINE_ROWS=$(( $(wc -l < "$REFINE_MANIFEST") - 1 ))
MISSING_NODES=$(hpc/python.sh -c \
  'import csv,sys; print(len({(r["alpha"],r["theta"],r["aspect_ratio"]) for r in csv.DictReader(open(sys.argv[1]))}))' \
  "$MISSING_MANIFEST")
COMBINED_ROWS=$(( $(wc -l < "$COMBINED_MANIFEST") - 1 ))
if (( MISSING_ROWS != 11664 || REFINE_ROWS != 1296 || MISSING_NODES != 108 || COMBINED_ROWS != 15552 )); then
  echo "unexpected correction design: missing=$MISSING_ROWS/$MISSING_NODES refinement=$REFINE_ROWS combined=$COMBINED_ROWS" >&2
  exit 2
fi
mkdir -p "$MISSING_RESULTS" "$REFINE_RESULTS" "$PRECOMPUTED" "$ARTIFACT_DIR"

PROP_CONCURRENT=18
REFINE_NODES=36

# Slurm job-array elements count toward association submission limits.  Use
# at most 256 long-lived workers and distribute rows by stride; this keeps all
# purchased cores occupied without injecting ~13,000 pending jobs at once.
submit_fit_workers() {
  local manifest=$1 rows=$2 dependency=${3:-}
  if (( rows == 0 )); then
    echo ""
    return
  fi
  local workers=${EXCITATION_MAX_CORES:-256}
  (( workers > rows )) && workers=$rows
  local dependency_args=()
  [[ -n "$dependency" ]] && dependency_args+=(--dependency="afterok:$dependency")
  local raw
  raw=$(sbatch --parsable --kill-on-invalid-dep=yes \
    "${dependency_args[@]}" --time="${EXCITATION_FIT_TIME:-14-00:00:00}" \
    --array="0-$((workers - 1))%$workers" \
    --export="ALL,EXCITATION_BOOTSTRAP=50,EXCITATION_OFFSETS=128" \
    hpc/excitation_fit_stride.slurm "$manifest")
  echo "${raw%%;*}"
}

if [[ "$RESUME_FROM_MISSING" == "true" ]]; then
  PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
    hpc/python.sh hpc/filter_missing_excitation.py \
    --manifest "$MISSING_MANIFEST" --output "$MISSING_RETRY_MANIFEST"
  RETRY_ROWS=$(( $(wc -l < "$MISSING_RETRY_MANIFEST") - 1 ))
  MISSING_SUBMITTED_ROWS=$RETRY_ROWS
  MISSING_FIT_JOBS=$(submit_fit_workers "$MISSING_RETRY_MANIFEST" "$RETRY_ROWS")
  PROP_JOB="reused"
else
  MISSING_SUBMITTED_ROWS=$MISSING_ROWS
  PROP_RAW=$(sbatch --parsable --array="0-$((MISSING_NODES - 1))%$PROP_CONCURRENT" \
    hpc/excitation_propensity_array.slurm "$MISSING_MANIFEST")
  PROP_JOB=${PROP_RAW%%;*}
  MISSING_FIT_JOBS=$(submit_fit_workers "$MISSING_MANIFEST" "$MISSING_ROWS" "$PROP_JOB")
fi
REFINE_PROP_RAW=$(sbatch --parsable --array="0-$((REFINE_NODES - 1))%3" \
  hpc/excitation_propensity_array.slurm "$REFINE_MANIFEST")
REFINE_PROP_JOB=${REFINE_PROP_RAW%%;*}
REFINE_FIT_JOBS=$(submit_fit_workers "$REFINE_MANIFEST" "$REFINE_ROWS" "$REFINE_PROP_JOB")
ALL_FIT_JOBS=$REFINE_FIT_JOBS
[[ -n "$MISSING_FIT_JOBS" ]] && ALL_FIT_JOBS="$MISSING_FIT_JOBS:$ALL_FIT_JOBS"
FULL_QA_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$ALL_FIT_JOBS" hpc/validate_excitation_surface.slurm \
  "$COMBINED_MANIFEST" "$COMBINED_SUMMARY" "$COMBINED_OFFLINE")
FULL_QA_JOB=${FULL_QA_RAW%%;*}
COEFFICIENT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$FULL_QA_JOB" hpc/compile_correction_surface.slurm \
  "$COEFFICIENT_ROWS" "$BASELINES" "$OLD_RESULTS" "$EXT_RESULTS" \
  "$REFINE_RESULTS" "$MISSING_RESULTS")
COEFFICIENT_JOB=${COEFFICIENT_RAW%%;*}

BASELINE_ROWS=$(( $(wc -l < "$GRID") - 1 ))
PRE_TASKS=$BASELINE_ROWS
PRE_CONCURRENT=$(( ${ARTIFACT_MAX_CORES:-256} / 2 )); (( PRE_CONCURRENT > PRE_TASKS )) && PRE_CONCURRENT=$PRE_TASKS
PRE_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$COEFFICIENT_JOB" \
  --array="0-$((PRE_TASKS - 1))%$PRE_CONCURRENT" \
  --export="ALL,COEFFICIENT_ROWS=$COEFFICIENT_ROWS" \
  hpc/artifact_precompute_stride.slurm "$GRID" "$BASELINES" "$PRECOMPUTED" \
  "$BASELINE_ROWS" "$OLD_RESULTS" "$EXT_RESULTS" "$REFINE_RESULTS" \
  "$MISSING_RESULTS")
PRE_JOB=${PRE_RAW%%;*}
PACK_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$PRE_JOB" \
  --export="ALL,COEFFICIENT_ROWS=$COEFFICIENT_ROWS" \
  hpc/aggregate.slurm "$GRID" "$BASELINES" "$ARTIFACT_DIR" "$PRECOMPUTED" \
  "$OLD_RESULTS" "$EXT_RESULTS" "$REFINE_RESULTS" "$MISSING_RESULTS")
PACK_JOB=${PACK_RAW%%;*}
PREP_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$PACK_JOB" \
  hpc/prepare_full_candidate_validation.slurm "$ARTIFACT" "$TAG")
PREP_JOB=${PREP_RAW%%;*}

# The 168 exact and 168 optimized runs cover all 28 artifact alpha/AR nodes,
# both initial partitions, and three replicates.  Capping each arm at 128 uses
# at most all 256 account cores.  The optimized arm is also the full-domain HCS
# gate, avoiding a third duplicate campaign.
HCS_REF_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$PREP_JOB" \
  --array=0-167%128 hpc/hcs_validation_array.slurm "$HCS_REF_MANIFEST" "$ARTIFACT" true)
HCS_REF_JOB=${HCS_REF_RAW%%;*}
HCS_OPT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$PREP_JOB" \
  --array=0-167%128 hpc/hcs_validation_array.slurm "$HCS_OPT_MANIFEST" "$ARTIFACT" true)
HCS_OPT_JOB=${HCS_OPT_RAW%%;*}
CADENCE_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HCS_REF_JOB:$HCS_OPT_JOB" hpc/compare_hcs_cadence.slurm \
  "$HCS_REF_MANIFEST" "$HCS_OPT_MANIFEST" "$CADENCE_RESULT")
CADENCE_JOB=${CADENCE_RAW%%;*}
HCS_PLOT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HCS_OPT_JOB" hpc/plot_hcs_validation.slurm \
  "$HCS_OPT_MANIFEST" "$HCS_FULL_RESULTS/hcs_theta_attraction.png" "$HCS_FULL_SUMMARY")
HCS_PLOT_JOB=${HCS_PLOT_RAW%%;*}

HOLDOUT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$PACK_JOB" \
  hpc/validate_independent_ctc_holdout.slurm "$HOLDOUT_MANIFEST" \
  "$HOLDOUT_BASELINE" "$ARTIFACT" "$HOLDOUT_VALIDATION")
HOLDOUT_JOB=${HOLDOUT_RAW%%;*}
USF_GATE_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HCS_PLOT_JOB:$CADENCE_JOB:$HOLDOUT_JOB" \
  hpc/check_usf_candidate.slurm "$HCS_FULL_SUMMARY" "$HOLDOUT_VALIDATION" \
  "$ARTIFACT" true)
USF_GATE_JOB=${USF_GATE_RAW%%;*}
USF_PILOT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$USF_GATE_JOB" --array=0-71%72 \
  hpc/usf_validation_array.slurm "$USF_PILOT_MANIFEST" "$ARTIFACT")
USF_PILOT_JOB=${USF_PILOT_RAW%%;*}
USF_PILOT_QA_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$USF_PILOT_JOB" hpc/analyze_usf_validation.slurm \
  "$USF_PILOT_MANIFEST" "$USF_PILOT_RESULTS/summary.json" \
  "$USF_PILOT_RESULTS/usf_validation.png")
USF_PILOT_QA_JOB=${USF_PILOT_QA_RAW%%;*}
USF_FULL_GATE_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$USF_PILOT_QA_JOB" hpc/check_usf_full.slurm \
  "$HCS_FULL_SUMMARY" "$HOLDOUT_VALIDATION" "$ARTIFACT" \
  "$USF_PILOT_RESULTS/summary.json")
USF_FULL_GATE_JOB=${USF_FULL_GATE_RAW%%;*}
USF_FULL_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$USF_FULL_GATE_JOB" --array=0-279%256 \
  hpc/usf_validation_array.slurm "$USF_FULL_MANIFEST" "$ARTIFACT")
USF_FULL_JOB=${USF_FULL_RAW%%;*}
USF_FULL_QA_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$USF_FULL_JOB" hpc/analyze_usf_validation.slurm \
  "$USF_FULL_MANIFEST" "$USF_FULL_RESULTS/summary.json" \
  "$USF_FULL_RESULTS/usf_validation.png")
USF_FULL_QA_JOB=${USF_FULL_QA_RAW%%;*}

NG_GATE_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HCS_PLOT_JOB:$CADENCE_JOB" hpc/check_hcs_ng.slurm \
  "$NG_MANIFEST" "$ARTIFACT" "$HCS_FULL_SUMMARY")
NG_GATE_JOB=${NG_GATE_RAW%%;*}
NG_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$NG_GATE_JOB" \
  --array=0-111%112 --export=ALL,HCS_NG_STRIDE=112 \
  hpc/hcs_ng_array.slurm "$NG_MANIFEST" "$ARTIFACT")
NG_JOB=${NG_RAW%%;*}
NG_QA_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$NG_JOB" \
  hpc/analyze_hcs_ng.slurm "$NG_MANIFEST" "$NG_RESULTS/summary.json" \
  "$NG_RESULTS/non_gaussian_observables.png")
NG_QA_JOB=${NG_QA_RAW%%;*}

echo "missing_correction_propensity_job=$PROP_JOB ($MISSING_NODES nodes; $PROP_CONCURRENT x 12 cores)"
echo "support_refinement_propensity_job=$REFINE_PROP_JOB (cache verification for 36 completed nodes)"
echo "missing_correction_fit_job=${MISSING_FIT_JOBS:-none} ($MISSING_SUBMITTED_ROWS rows submitted; $MISSING_ROWS rows in complete design; distributed over at most ${EXCITATION_MAX_CORES:-256} workers)"
echo "support_refinement_fit_job=$REFINE_FIT_JOBS ($REFINE_ROWS rows distributed over at most ${EXCITATION_MAX_CORES:-256} workers)"
echo "combined_correction_QA_job=$FULL_QA_JOB"
echo "coefficient_surface_job=$COEFFICIENT_JOB (compiled once, shared by all artifact tasks)"
echo "artifact_precompute_job=$PRE_JOB ($PRE_CONCURRENT x 2 cores); pack=$PACK_JOB; prepare=$PREP_JOB"
echo "HCS_full_domain_reference=$HCS_REF_JOB; HCS_full_domain_optimized=$HCS_OPT_JOB"
echo "HCS_cadence_QA=$CADENCE_JOB; HCS_full_summary=$HCS_PLOT_JOB; fresh_CTC_rescore=$HOLDOUT_JOB"
echo "USF_pilot=$USF_PILOT_JOB (72 tasks); pilot_QA=$USF_PILOT_QA_JOB; conditional_full_USF=$USF_FULL_JOB (280 tasks); full_QA=$USF_FULL_QA_JOB"
echo "nonGaussian_domain_pilot=$NG_JOB; nonGaussian_QA=$NG_QA_JOB"
echo "No new CTC trajectory is submitted; all response fits reuse the frozen 128-offset caches."
