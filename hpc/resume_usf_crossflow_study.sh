#!/bin/bash
# Resume a staged USF campaign without rerunning complete realizations.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${1:-usf_crossflow_v1_20260927}
ARTIFACT=${2:-models/microscopic_closure_v2_angular_evidence/closure_v2.npz}
HCS_SUMMARY=${3:-results/hcs_ng_sweep_angular_v8/summary.json}
ATTEMPT=${USF_RESUME_ATTEMPT:-1}
REFERENCE_PROVENANCE=DSMC_0D_v2/reference/usf_reference_provenance_v1.json
MAX_CORES=${USF_MAX_CORES:-256}
MEMORY=${USF_MEM_PER_TASK:-1800M}
WALLTIME=${USF_WALLTIME:-08:00:00}
QOS=${USF_QOS:-normal}
PROMOTED_DIR="models/microscopic_closure_v2_usf_validated_${TAG}"
RECEIPT="results/usf_${TAG}_resume_${ATTEMPT}.txt"

if [[ -e "$RECEIPT" ]]; then
  echo "resume receipt already exists: $RECEIPT" >&2
  exit 2
fi
if (( MAX_CORES < 1 || MAX_CORES > 256 )); then
  echo "USF_MAX_CORES must lie in [1,256] for the morri353 allocation" >&2
  exit 2
fi
if [[ -d "$PROMOTED_DIR" ]] && compgen -G "$PROMOTED_DIR/*" >/dev/null; then
  echo "refusing to overwrite promoted artifact directory $PROMOTED_DIR" >&2
  exit 2
fi

mkdir -p logs manifests
declare -A MANIFEST RETRY REPORT RESULT SUMMARY FIGURE ROWS ARRAY QA GATE
for MODE in numerics pilot full; do
  MANIFEST[$MODE]="manifests/usf_${TAG}_${MODE}.csv"
  RETRY[$MODE]="manifests/usf_${TAG}_${MODE}_retry_${ATTEMPT}.csv"
  REPORT[$MODE]="results/usf_${TAG}_${MODE}_retry_${ATTEMPT}.json"
  RESULT[$MODE]="results/usf_${TAG}_${MODE}"
  SUMMARY[$MODE]="${RESULT[$MODE]}/summary.json"
  FIGURE[$MODE]="${RESULT[$MODE]}/usf_crossflow.png"
  if [[ ! -f "${MANIFEST[$MODE]}" ]]; then
    echo "missing original campaign manifest: ${MANIFEST[$MODE]}" >&2
    exit 2
  fi
  hpc/python.sh DSMC_0D_v2/scripts/make_usf_retry_manifest.py \
    --manifest "${MANIFEST[$MODE]}" --output "${RETRY[$MODE]}" \
    --report "${REPORT[$MODE]}"
  ROWS[$MODE]=$(( $(wc -l < "${RETRY[$MODE]}") - 1 ))
done

PYTHONPATH="$ROOT/contracts/python:$ROOT/DSMC_0D_v2/src" hpc/python.sh \
  hpc/require_usf_study_prerequisites.py \
    --artifact "$ARTIFACT" --hcs-summary "$HCS_SUMMARY" \
    --reference-provenance "$REFERENCE_PROVENANCE" \
    --numerics-manifest "${MANIFEST[numerics]}" \
    --pilot-manifest "${MANIFEST[pilot]}" \
    --full-manifest "${MANIFEST[full]}"

submit_array() {
  local mode=$1 upstream=${2:-} tasks concurrency raw
  tasks=${ROWS[$mode]}
  if (( tasks == 0 )); then
    printf '%s' "reused"
    return
  fi
  concurrency=$MAX_CORES
  (( concurrency > tasks )) && concurrency=$tasks
  local args=(--parsable --kill-on-invalid-dep=yes
    --array="0-$((tasks - 1))%${concurrency}" --mem="$MEMORY"
    --time="$WALLTIME" --qos="$QOS"
    --export="ALL,USF_STUDY_STRIDE=$tasks")
  [[ -n "$upstream" ]] && args+=(--dependency="afterok:$upstream")
  raw=$(sbatch "${args[@]}" hpc/usf_study_array.slurm \
    "${RETRY[$mode]}" "$ARTIFACT")
  printf '%s' "${raw%%;*}"
}

submit_analysis() {
  local mode=$1 dependency=${2:-} raw
  local args=(--parsable --kill-on-invalid-dep=yes)
  [[ -n "$dependency" ]] && args+=(--dependency="$dependency")
  raw=$(sbatch "${args[@]}" hpc/analyze_usf_study.slurm \
    "${MANIFEST[$mode]}" "${SUMMARY[$mode]}" "${FIGURE[$mode]}")
  printf '%s' "${raw%%;*}"
}

submit_gate() {
  local mode=$1 analysis_job=$2 raw
  raw=$(sbatch --parsable --kill-on-invalid-dep=yes \
    --dependency="afterok:$analysis_job" hpc/check_usf_study_stage.slurm \
    "${SUMMARY[$mode]}" "$mode")
  printf '%s' "${raw%%;*}"
}

submit_stage() {
  local mode=$1 upstream=${2:-} analysis_dependency
  ARRAY[$mode]=$(submit_array "$mode" "$upstream")
  if [[ "${ARRAY[$mode]}" == "reused" ]]; then
    analysis_dependency=""
    [[ -n "$upstream" ]] && analysis_dependency="afterok:$upstream"
  else
    analysis_dependency="afterany:${ARRAY[$mode]}"
  fi
  QA[$mode]=$(submit_analysis "$mode" "$analysis_dependency")
  GATE[$mode]=$(submit_gate "$mode" "${QA[$mode]}")
}

submit_stage numerics
submit_stage pilot "${GATE[numerics]}"
submit_stage full "${GATE[pilot]}"
PROMOTION_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:${GATE[full]}" hpc/promote_usf_validated_artifact.slurm \
  "$ARTIFACT" "$HCS_SUMMARY" "${SUMMARY[full]}" "$PROMOTED_DIR")
PROMOTION_JOB=${PROMOTION_RAW%%;*}

{
  printf "attempt=%s\n" "$ATTEMPT"
  for MODE in numerics pilot full; do
    printf "%s_retry_rows=%s\n" "$MODE" "${ROWS[$MODE]}"
    printf "%s_array=%s\n" "$MODE" "${ARRAY[$MODE]}"
    printf "%s_qa=%s\n" "$MODE" "${QA[$MODE]}"
    printf "%s_gate=%s\n" "$MODE" "${GATE[$MODE]}"
  done
  printf "promotion=%s\n" "$PROMOTION_JOB"
} > "$RECEIPT"

for MODE in numerics pilot full; do
  echo "$MODE: retry_rows=${ROWS[$MODE]} array=${ARRAY[$MODE]} qa=${QA[$MODE]} gate=${GATE[$MODE]}"
done
echo "promotion=$PROMOTION_JOB output=$PROMOTED_DIR"
echo "allocation: concurrency=$MAX_CORES one-core realizations, memory/task=$MEMORY, walltime=$WALLTIME, qos=$QOS"
echo "Completed task outputs are reused; only retry manifests enter new arrays."
echo "Monitor: squeue -u \"$USER\" -o '%.18i %.24j %.2t %.10M %.10l %.6D %R'"
echo "Final verdict: ${SUMMARY[full]}"
