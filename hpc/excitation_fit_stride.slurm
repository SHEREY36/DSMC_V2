#!/bin/bash -l
# Independent virtual excitation fits.  No fresh CTC trajectory is generated.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=excitation_fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/excitation_fit_%A_%a.out
#SBATCH --error=logs/excitation_fit_%A_%a.err

set -euo pipefail
MANIFEST=${1:?excitation manifest is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
STRIDE=$(( ${SLURM_ARRAY_TASK_MAX:?} - ${SLURM_ARRAY_TASK_MIN:?} + 1 ))
INDEX=${SLURM_ARRAY_TASK_ID:?}
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))

while (( INDEX < ROWS )); do
  PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
    hpc/python.sh hpc/run_excitation_task.py \
      --manifest "$MANIFEST" --index "$INDEX" \
      --bootstrap "${EXCITATION_BOOTSTRAP:-50}" \
      --propensity-offsets "${EXCITATION_OFFSETS:-128}"
  INDEX=$(( INDEX + STRIDE ))
done
