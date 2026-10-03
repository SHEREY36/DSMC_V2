#!/bin/bash
# Refit only the rejected nodes, revalidate all estimates, then build.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
GRID=${1:-manifests/artifact_grid.csv}
ESTIMATES=${2:-results/closure_estimates/artifact_grid}
OUTPUT=${3:-models/microscopic_closure_v2}
SOURCE_REPORT=${4:-results/closure_estimates/artifact_grid_validation.json}
REPAIRS=${5:-manifests/artifact_grid_repairs.csv}
REPAIR_REPORT=${6:-results/closure_estimates/artifact_grid_repair_validation.json}
WORK=${7:-results/closure_estimates/artifact_precompute}

mkdir -p logs "$WORK"

hpc/python.sh hpc/make_artifact_repair_manifest.py \
  --grid "$GRID" --validation "$SOURCE_REPORT" --output "$REPAIRS"
ROWS=$(( $(wc -l < "$REPAIRS") - 1 ))
if (( ROWS <= 0 )); then
  echo "No failed model-form nodes were selected; refusing an empty array." >&2
  exit 2
fi
MAX_CONCURRENT=${CLOSURE_MAX_CONCURRENT:-32}
FIT_JOB_RAW=$(sbatch --parsable --array="0-$((ROWS - 1))%$MAX_CONCURRENT" \
  hpc/closure_fit_array.slurm "$REPAIRS" "$ESTIMATES")
FIT_JOB=${FIT_JOB_RAW%%;*}
QA_JOB_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$FIT_JOB" \
  hpc/validate_artifact_estimates.slurm "$GRID" "$ESTIMATES" "$REPAIR_REPORT")
QA_JOB=${QA_JOB_RAW%%;*}
ARTIFACT_ROWS=$(( $(wc -l < "$GRID") - 1 ))
MAX_ARRAY=$(scontrol show config 2>/dev/null \
  | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
MAX_ARRAY=${MAX_ARRAY:-1000}
PRE_TASKS=$ARTIFACT_ROWS
(( PRE_TASKS > MAX_ARRAY )) && PRE_TASKS=$MAX_ARRAY
PRE_CONCURRENT=$(( ${ARTIFACT_MAX_CORES:-256} / 2 ))
(( PRE_CONCURRENT > PRE_TASKS )) && PRE_CONCURRENT=$PRE_TASKS
PRE_JOB_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$QA_JOB" \
  --array="0-$((PRE_TASKS - 1))%$PRE_CONCURRENT" \
  hpc/artifact_precompute_stride.slurm \
  "$GRID" "$ESTIMATES" "$WORK" "$ARTIFACT_ROWS")
PRE_JOB=${PRE_JOB_RAW%%;*}
BUILD_JOB_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PRE_JOB" \
  hpc/aggregate.slurm "$GRID" "$ESTIMATES" "$OUTPUT" "$WORK")
BUILD_JOB=${BUILD_JOB_RAW%%;*}

echo "repair_fit_job=$FIT_JOB"
echo "estimate_qa_job=$QA_JOB"
echo "precompute_job=$PRE_JOB (${PRE_TASKS} work queues; ${PRE_CONCURRENT} simultaneous x 2 cores)"
echo "artifact_job=$BUILD_JOB"
echo "Only the nodes rejected by the copied validation report are refitted."
