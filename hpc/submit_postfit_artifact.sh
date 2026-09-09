#!/bin/bash
# Reuse the already passing 144 fits and submit only the deployable artifact build.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
GRID=${1:-manifests/artifact_grid.csv}
ESTIMATES=${2:-results/closure_estimates/artifact_grid}
ARTIFACT_DIR=${3:-models/microscopic_closure_v2}

mkdir -p logs "$ARTIFACT_DIR" results/closure_estimates
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/validate_artifact_inputs.py \
  --manifest "$GRID" --estimates "$ESTIMATES" \
  --output results/closure_estimates/artifact_grid_postfit_validation.json \
  --require-current-estimates --require-pass

JOB_RAW=$(sbatch --parsable hpc/aggregate.slurm "$GRID" "$ESTIMATES" "$ARTIFACT_DIR")
JOB=${JOB_RAW%%;*}
echo "artifact_job=$JOB"
echo "Only the artifact build was submitted; no HCS, CTC, or closure-fit job was submitted."
echo "After completion inspect: logs/artifact_${JOB}.{out,err} and $ARTIFACT_DIR/manifest.json"
