#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_study
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Measured HCS-v8 peak RSS was about 0.64 GiB/task.  This leaves substantial
# headroom while allowing 128 independent realizations on each 256-GiB node.
#SBATCH --mem=1800M
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/usf_study_%A_%a.out
#SBATCH --error=logs/usf_study_%A_%a.err

set -euo pipefail
MANIFEST=${1:?manifest required}
ARTIFACT=${2:?artifact required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
STRIDE=${USF_STUDY_STRIDE:-$(( ${SLURM_ARRAY_TASK_MAX:?} - ${SLURM_ARRAY_TASK_MIN:?} + 1 ))}
INDEX=${SLURM_ARRAY_TASK_ID:?}
while (( INDEX < ROWS )); do
  PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src" \
    hpc/python.sh DSMC_0D_v2/scripts/run_usf_study_task.py \
      --manifest "$MANIFEST" --task "$INDEX" --artifact "$ARTIFACT"
  INDEX=$(( INDEX + STRIDE ))
done
