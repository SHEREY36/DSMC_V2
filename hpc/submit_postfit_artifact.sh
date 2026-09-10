#!/bin/bash
# Reuse the already passing 144 fits and submit only the deployable artifact build.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
GRID=${1:-manifests/artifact_grid.csv}
ESTIMATES=${2:-results/closure_estimates/artifact_grid}
ARTIFACT_DIR=${3:-models/microscopic_closure_v2}
WORK=${4:-results/closure_estimates/artifact_precompute}

mkdir -p logs "$ARTIFACT_DIR" "$WORK" results/closure_estimates
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/validate_artifact_inputs.py \
  --manifest "$GRID" --estimates "$ESTIMATES" \
  --output results/closure_estimates/artifact_grid_postfit_validation.json \
  --require-current-estimates --require-pass

ROWS=$(( $(wc -l < "$GRID") - 1 ))
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk '/MaxArraySize/ {print $3; exit}')
MAX_ARRAY=${MAX_ARRAY:-1000}
MAX_CORES=${ARTIFACT_MAX_CORES:-256}
CPUS_PER_TASK=2
ARRAY_TASKS=$ROWS
(( ARRAY_TASKS > MAX_ARRAY )) && ARRAY_TASKS=$MAX_ARRAY
CONCURRENT=$(( MAX_CORES / CPUS_PER_TASK ))
(( CONCURRENT > ARRAY_TASKS )) && CONCURRENT=$ARRAY_TASKS
if (( ARRAY_TASKS < 1 || CONCURRENT < 1 )); then
  echo "No Slurm array capacity available for artifact precompute" >&2
  exit 2
fi

PRE_RAW=$(sbatch --parsable --array="0-$((ARRAY_TASKS - 1))%$CONCURRENT" \
  hpc/artifact_precompute_stride.slurm "$GRID" "$ESTIMATES" "$WORK" "$ROWS")
PRE_JOB=${PRE_RAW%%;*}
JOB_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PRE_JOB" \
  hpc/aggregate.slurm "$GRID" "$ESTIMATES" "$ARTIFACT_DIR" "$WORK")
JOB=${JOB_RAW%%;*}
echo "precompute_job=$PRE_JOB (${ARRAY_TASKS} work queues; ${CONCURRENT} simultaneous x ${CPUS_PER_TASK} cores)"
echo "artifact_job=$JOB (afterok:$PRE_JOB)"
echo "No HCS, CTC, or closure-fit job was submitted."
echo "After completion inspect: logs/artifact_${JOB}.{out,err} and $ARTIFACT_DIR/manifest.json"
