#!/bin/bash
# Submit one continuation job after all currently accepted fit arrays.

set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p logs

mapfile -t ACTIVE < <(
  squeue -h -u "$USER" -n excitation -o "%A" | sort -u
)
DEPENDENCY_ARGS=()
if (( ${#ACTIVE[@]} > 0 )); then
  DEPENDENCY=$(IFS=:; echo "${ACTIVE[*]}")
  DEPENDENCY_ARGS+=(--dependency="afterany:$DEPENDENCY")
  echo "resume will wait for ${#ACTIVE[@]} active excitation arrays: $DEPENDENCY"
else
  echo "no active excitation arrays; resume may start immediately"
fi

RAW=$(sbatch --parsable --kill-on-invalid-dep=yes "${DEPENDENCY_ARGS[@]}" \
  --export="ALL,TAG=${TAG:-full_candidate_v1},EXCITATION_MAX_CORES=${EXCITATION_MAX_CORES:-256},ARTIFACT_MAX_CORES=${ARTIFACT_MAX_CORES:-256}" \
  hpc/resume_full_model_pipeline.slurm)
echo "full_model_resume_job=${RAW%%;*}"
