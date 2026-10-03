#!/bin/bash -l
# One deterministic geometry/acceptance calculation per physical CTC node.
# Every one of the node's 72 response fits reuses the resulting cache.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=exc_propensity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exc_propensity_%A_%a.out
#SBATCH --error=logs/exc_propensity_%A_%a.err

set -euo pipefail
MANIFEST=${1:?excitation manifest is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
hpc/python.sh hpc/precompute_excitation_propensity.py \
  --manifest "$MANIFEST" --index "${SLURM_ARRAY_TASK_ID:?}" \
  --offsets "${EXCITATION_OFFSETS:-128}" \
  --workers "${SLURM_CPUS_PER_TASK:-12}"
