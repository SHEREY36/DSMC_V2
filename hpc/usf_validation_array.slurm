#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=usf_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=24:00:00
#SBATCH --output=logs/usf_gate_%A_%a.out
#SBATCH --error=logs/usf_gate_%A_%a.err

set -euo pipefail
MANIFEST=${1:?USF manifest is required}
ARTIFACT=${2:?candidate artifact is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
hpc/python.sh DSMC_0D_v2/scripts/run_usf_validation_task.py \
  --manifest "$MANIFEST" --task "$SLURM_ARRAY_TASK_ID" \
  --artifact "$ARTIFACT"
