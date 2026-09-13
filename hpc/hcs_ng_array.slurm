#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=hcs_ng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/hcs_ng_%A_%a.out
#SBATCH --error=logs/hcs_ng_%A_%a.err

set -euo pipefail
MANIFEST=${1:?manifest required}
ARTIFACT=${2:?artifact required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
STRIDE=${HCS_NG_STRIDE:-$(( ${SLURM_ARRAY_TASK_MAX:?} - ${SLURM_ARRAY_TASK_MIN:?} + 1 ))}
INDEX=${SLURM_ARRAY_TASK_ID:?}
while (( INDEX < ROWS )); do
  PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src" \
    hpc/python.sh DSMC_0D_v2/scripts/run_hcs_ng_task.py \
      --manifest "$MANIFEST" --task "$INDEX" --artifact "$ARTIFACT"
  INDEX=$(( INDEX + STRIDE ))
done
