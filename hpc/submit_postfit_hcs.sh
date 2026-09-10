#!/bin/bash
# Reuse the already passing 144 fits: validate -> build artifact -> HCS gate -> plot.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
GRID=${1:-manifests/artifact_grid.csv}
ESTIMATES=${2:-results/closure_estimates/artifact_grid}
ARTIFACT_DIR=${3:-models/microscopic_closure_v2}
HCS_MANIFEST=${4:-manifests/hcs_validation.csv}
WORK=${5:-results/closure_estimates/artifact_precompute}

mkdir -p logs "$ARTIFACT_DIR" "$WORK" results/hcs_validation
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/validate_artifact_inputs.py \
  --manifest "$GRID" --estimates "$ESTIMATES" \
  --output results/closure_estimates/artifact_grid_postfit_validation.json \
  --require-current-estimates --require-pass

ARTIFACT_ROWS=$(( $(wc -l < "$GRID") - 1 ))
MAX_ARRAY=$(scontrol show config 2>/dev/null \
  | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
MAX_ARRAY=${MAX_ARRAY:-1000}
MAX_CORES=${ARTIFACT_MAX_CORES:-256}
PRE_TASKS=$ARTIFACT_ROWS
(( PRE_TASKS > MAX_ARRAY )) && PRE_TASKS=$MAX_ARRAY
PRE_CONCURRENT=$(( MAX_CORES / 2 ))
(( PRE_CONCURRENT > PRE_TASKS )) && PRE_CONCURRENT=$PRE_TASKS
PRE_RAW=$(sbatch --parsable --array="0-$((PRE_TASKS - 1))%$PRE_CONCURRENT" \
  hpc/artifact_precompute_stride.slurm \
  "$GRID" "$ESTIMATES" "$WORK" "$ARTIFACT_ROWS")
PRE_JOB=${PRE_RAW%%;*}
ARTIFACT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PRE_JOB" \
  hpc/aggregate.slurm "$GRID" "$ESTIMATES" "$ARTIFACT_DIR" "$WORK")
ARTIFACT_JOB=${ARTIFACT_RAW%%;*}
hpc/python.sh DSMC_0D_v2/scripts/make_hcs_validation_manifest.py --output "$HCS_MANIFEST"
ROWS=$(( $(wc -l < "$HCS_MANIFEST") - 1 ))
HCS_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$ARTIFACT_JOB" --array="0-$((ROWS - 1))%20" \
  hpc/hcs_validation_array.slurm "$HCS_MANIFEST" "$ARTIFACT_DIR/closure_v2.npz")
HCS_JOB=${HCS_RAW%%;*}
PLOT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HCS_JOB" hpc/plot_hcs_validation.slurm "$HCS_MANIFEST")
PLOT_JOB=${PLOT_RAW%%;*}

echo "precompute_job=$PRE_JOB (${PRE_TASKS} work queues; ${PRE_CONCURRENT} simultaneous x 2 cores)"
echo "artifact_job=$ARTIFACT_JOB (afterok:$PRE_JOB)"
echo "hcs_job=$HCS_JOB"
echo "plot_job=$PLOT_JOB"
echo "No CTC or closure refits were resubmitted."
