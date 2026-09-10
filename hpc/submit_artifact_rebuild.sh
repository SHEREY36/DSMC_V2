#!/bin/bash
# Submit the canonical refit -> deep QA -> fail-closed artifact rebuild chain.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MANIFEST=${1:-manifests/artifact_grid.csv}
ESTIMATES=${2:-results/closure_estimates/artifact_grid}
OUTPUT=${3:-models/microscopic_closure_v2}
REPORT=${4:-results/closure_estimates/artifact_grid_validation.json}
WORK=${5:-results/closure_estimates/artifact_precompute}

mkdir -p logs "$ESTIMATES" "$WORK" "$(dirname "$REPORT")"
hpc/python.sh hpc/make_artifact_manifest.py \
  manifests/closure_sentinel.csv \
  manifests/ar_extension.csv \
  manifests/ar_near_sphere.csv \
  manifests/ar_low_theta.csv \
  --output "$MANIFEST" --require-complete

ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
if (( ROWS < 1 )); then
  echo "canonical artifact manifest is empty" >&2
  exit 2
fi
MAX_CONCURRENT=${CLOSURE_MAX_CONCURRENT:-256}
FIT_JOB_RAW=$(sbatch --parsable --array="0-$((ROWS - 1))%$MAX_CONCURRENT" \
  hpc/closure_fit_array.slurm "$MANIFEST" "$ESTIMATES")
FIT_JOB=${FIT_JOB_RAW%%;*}
QA_JOB_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$FIT_JOB" \
  hpc/validate_artifact_grid.slurm "$MANIFEST" "$ESTIMATES" "$REPORT")
QA_JOB=${QA_JOB_RAW%%;*}
MAX_ARRAY=$(scontrol show config 2>/dev/null \
  | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
MAX_ARRAY=${MAX_ARRAY:-1000}
PRE_TASKS=$ROWS
(( PRE_TASKS > MAX_ARRAY )) && PRE_TASKS=$MAX_ARRAY
PRE_CONCURRENT=$(( ${ARTIFACT_MAX_CORES:-256} / 2 ))
(( PRE_CONCURRENT > PRE_TASKS )) && PRE_CONCURRENT=$PRE_TASKS
PRE_JOB_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$QA_JOB" \
  --array="0-$((PRE_TASKS - 1))%$PRE_CONCURRENT" \
  hpc/artifact_precompute_stride.slurm \
  "$MANIFEST" "$ESTIMATES" "$WORK" "$ROWS")
PRE_JOB=${PRE_JOB_RAW%%;*}
BUILD_JOB_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PRE_JOB" \
  hpc/aggregate.slurm "$MANIFEST" "$ESTIMATES" "$OUTPUT" "$WORK")
BUILD_JOB=${BUILD_JOB_RAW%%;*}

echo "fit_job=$FIT_JOB"
echo "qa_job=$QA_JOB"
echo "precompute_job=$PRE_JOB (${PRE_TASKS} work queues; ${PRE_CONCURRENT} simultaneous x 2 cores)"
echo "artifact_job=$BUILD_JOB"
echo "The artifact job runs only if all 144 current-contract estimates pass deep QA."
