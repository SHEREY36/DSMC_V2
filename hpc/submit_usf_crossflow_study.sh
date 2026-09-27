#!/bin/bash
# Submit one dependency-chained numerical -> pilot -> full USF study.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${1:-crossflow_v1}
ARTIFACT=${2:-models/microscopic_closure_v2_angular_evidence/closure_v2.npz}
HCS_SUMMARY=${3:-results/hcs_ng_sweep_angular_v8/summary.json}
REFERENCE_PROVENANCE=DSMC_0D_v2/reference/usf_reference_provenance_v1.json
SPHERE_REFERENCE=DSMC_0D_v2/reference/usf_sphere_boltzmann_v1.csv
DEM_REFERENCE=DSMC_0D_v2/reference/usf_dem_fresh_v1.csv
MAX_CORES=${USF_MAX_CORES:-256}
MEMORY=${USF_MEM_PER_TASK:-1800M}
WALLTIME=${USF_WALLTIME:-08:00:00}
QOS=${USF_QOS:-normal}
PROMOTED_DIR="models/microscopic_closure_v2_usf_validated_$TAG"
SUBMISSION_RECORD="results/usf_${TAG}_submission.txt"

if [[ -e "$SUBMISSION_RECORD" ]]; then
  echo "submission record already exists: $SUBMISSION_RECORD" >&2
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
declare -A MANIFEST RESULT SUMMARY FIGURE ROWS
for MODE in numerics pilot full; do
  MANIFEST[$MODE]="manifests/usf_${TAG}_${MODE}.csv"
  RESULT[$MODE]="results/usf_${TAG}_${MODE}"
  SUMMARY[$MODE]="${RESULT[$MODE]}/summary.json"
  FIGURE[$MODE]="${RESULT[$MODE]}/usf_crossflow.png"
  if [[ -e "${SUMMARY[$MODE]}" ]] || compgen -G "${RESULT[$MODE]}/*" >/dev/null; then
    echo "refusing to mix a new campaign with outputs in ${RESULT[$MODE]}" >&2
    exit 2
  fi
  mkdir -p "${RESULT[$MODE]}"
  PYTHONPATH="$ROOT/DSMC_0D_v2/src" hpc/python.sh \
    DSMC_0D_v2/scripts/make_usf_study_manifest.py \
      --mode "$MODE" --output "${MANIFEST[$MODE]}" \
      --results "${RESULT[$MODE]}" --reference "$DEM_REFERENCE" \
      --artifact "$ARTIFACT"
  ROWS[$MODE]=$(( $(wc -l < "${MANIFEST[$MODE]}") - 1 ))
done

PYTHONPATH="$ROOT/contracts/python:$ROOT/DSMC_0D_v2/src" hpc/python.sh \
  hpc/require_usf_study_prerequisites.py \
    --artifact "$ARTIFACT" --hcs-summary "$HCS_SUMMARY" \
    --reference-provenance "$REFERENCE_PROVENANCE" \
    --numerics-manifest "${MANIFEST[numerics]}" \
    --pilot-manifest "${MANIFEST[pilot]}" --full-manifest "${MANIFEST[full]}"

submit_array() {
  local mode=$1 dependency=${2:-} tasks concurrency raw
  tasks=${ROWS[$mode]}
  concurrency=$MAX_CORES; (( concurrency > tasks )) && concurrency=$tasks
  local args=(--parsable --kill-on-invalid-dep=yes
    --array="0-$((tasks - 1))%$concurrency" --mem="$MEMORY"
    --time="$WALLTIME" --qos="$QOS"
    --export="ALL,USF_STUDY_STRIDE=$tasks")
  [[ -n "$dependency" ]] && args+=(--dependency="afterok:$dependency")
  raw=$(sbatch "${args[@]}" hpc/usf_study_array.slurm \
    "${MANIFEST[$mode]}" "$ARTIFACT")
  printf '%s' "${raw%%;*}"
}

submit_analysis() {
  local mode=$1 array_job=$2 raw
  raw=$(sbatch --parsable --kill-on-invalid-dep=yes \
    --dependency="afterany:$array_job" hpc/analyze_usf_study.slurm \
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

NUMERICS_JOB=$(submit_array numerics)
NUMERICS_QA=$(submit_analysis numerics "$NUMERICS_JOB")
NUMERICS_GATE=$(submit_gate numerics "$NUMERICS_QA")
PILOT_JOB=$(submit_array pilot "$NUMERICS_GATE")
PILOT_QA=$(submit_analysis pilot "$PILOT_JOB")
PILOT_GATE=$(submit_gate pilot "$PILOT_QA")
FULL_JOB=$(submit_array full "$PILOT_GATE")
FULL_QA=$(submit_analysis full "$FULL_JOB")
FULL_GATE=$(submit_gate full "$FULL_QA")
PROMOTION_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$FULL_GATE" hpc/promote_usf_validated_artifact.slurm \
  "$ARTIFACT" "$HCS_SUMMARY" "${SUMMARY[full]}" "$PROMOTED_DIR")
PROMOTION_JOB=${PROMOTION_RAW%%;*}
{
  printf "numerics=%s\n" "$NUMERICS_JOB"
  printf "numerics_qa=%s\n" "$NUMERICS_QA"
  printf "numerics_gate=%s\n" "$NUMERICS_GATE"
  printf "pilot=%s\n" "$PILOT_JOB"
  printf "pilot_qa=%s\n" "$PILOT_QA"
  printf "pilot_gate=%s\n" "$PILOT_GATE"
  printf "full=%s\n" "$FULL_JOB"
  printf "full_qa=%s\n" "$FULL_QA"
  printf "full_gate=%s\n" "$FULL_GATE"
  printf "promotion=%s\n" "$PROMOTION_JOB"
} > "$SUBMISSION_RECORD"

echo "numerics=$NUMERICS_JOB (${ROWS[numerics]} runs) qa=$NUMERICS_QA gate=$NUMERICS_GATE"
echo "pilot=$PILOT_JOB (${ROWS[pilot]} runs) qa=$PILOT_QA gate=$PILOT_GATE"
echo "full=$FULL_JOB (${ROWS[full]} runs) qa=$FULL_QA gate=$FULL_GATE"
echo "promotion=$PROMOTION_JOB (afterok:$FULL_GATE) output=$PROMOTED_DIR"
echo "allocation: concurrency=$MAX_CORES one-core realizations, memory/task=$MEMORY, walltime=$WALLTIME, qos=$QOS"
echo "The full stage starts only if both smaller scientific gates pass."
echo "Monitor: squeue -u \"$USER\" -o '%.18i %.24j %.2t %.10M %.10l %.6D %R'"
echo "Final verdict: ${SUMMARY[full]}"
