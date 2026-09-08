#!/bin/bash
# Submit the canonical refit -> deep QA -> fail-closed artifact rebuild chain.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MANIFEST=${1:-manifests/artifact_grid.csv}
ESTIMATES=${2:-results/closure_estimates/artifact_grid}
OUTPUT=${3:-models/microscopic_closure_v2}
REPORT=${4:-results/closure_estimates/artifact_grid_validation.json}

mkdir -p logs "$ESTIMATES" "$(dirname "$REPORT")"
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
QA_JOB_RAW=$(sbatch --parsable --dependency="afterok:$FIT_JOB" \
  hpc/validate_artifact_grid.slurm "$MANIFEST" "$ESTIMATES" "$REPORT")
QA_JOB=${QA_JOB_RAW%%;*}
BUILD_JOB_RAW=$(sbatch --parsable --dependency="afterok:$QA_JOB" \
  hpc/aggregate.slurm "$MANIFEST" "$ESTIMATES" "$OUTPUT")
BUILD_JOB=${BUILD_JOB_RAW%%;*}

echo "fit_job=$FIT_JOB"
echo "qa_job=$QA_JOB"
echo "artifact_job=$BUILD_JOB"
echo "The artifact job runs only if all 144 current-contract estimates pass deep QA."
