#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_exact_fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
# Three CPUs remain the dominant TRES request at this memory size on Negishi.
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/usf_exact_fit_%A_%a.out
#SBATCH --error=logs/usf_exact_fit_%A_%a.err
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/estimate_usf_response_task.py \
  --manifest "${1:?CTC manifest}" --index "$SLURM_ARRAY_TASK_ID" \
  --output "${2:?estimate output}" --bootstrap "${RESPONSE_BOOTSTRAP:-100}" \
  --propensity-offsets "${RESPONSE_PROPENSITY_OFFSETS:-64}" \
  --propensity-workers "${SLURM_CPUS_PER_TASK:-3}"
