#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_harvest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/usf_harvest_%A_%a.out
#SBATCH --error=logs/usf_harvest_%A_%a.err
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src" \
  hpc/python.sh DSMC_0D_v2/scripts/run_usf_response_harvest.py \
  --manifest "${1:?manifest required}" --task "$SLURM_ARRAY_TASK_ID"
