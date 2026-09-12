#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=hcs_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hcs_gate_%A_%a.out
#SBATCH --error=logs/hcs_gate_%A_%a.err

set -euo pipefail
MANIFEST=${1:-manifests/hcs_validation.csv}
ARTIFACT=${2:-models/microscopic_closure_v2/closure_v2.npz}
CORRECTIONS=${3:-false}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OMP_NUM_THREADS=1
CORRECTION_ARGS=()
if [[ "$CORRECTIONS" == "true" ]]; then
  CORRECTION_ARGS+=(--enable-invariant-corrections)
fi
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src" \
  hpc/python.sh DSMC_0D_v2/scripts/run_hcs_validation_task.py \
  --manifest "$MANIFEST" --task "$SLURM_ARRAY_TASK_ID" --artifact "$ARTIFACT" \
  "${CORRECTION_ARGS[@]}"
