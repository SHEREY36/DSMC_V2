#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -J closure_fit_sentinel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH --array=0-35%12
#SBATCH -o logs/closure_fit_sentinel_%A_%a.out
#SBATCH -e logs/closure_fit_sentinel_%A_%a.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
if [[ ! -d "$ROOT/Coll_Models_v2" ]]; then
    printf 'ERROR: submit this file from the DSMC_V2 repository root.\n' >&2
    exit 2
fi
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
    hpc/python.sh hpc/estimate_closure_task.py \
    --manifest manifests/closure_sentinel.csv \
    --index "$SLURM_ARRAY_TASK_ID" \
    --output results/closure_estimates/sentinel --bootstrap 500
