#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -J closure_fit_baseline
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --array=0-1247%40
#SBATCH -o logs/closure_fit_baseline_%A_%a.out
#SBATCH -e logs/closure_fit_baseline_%A_%a.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
if [[ ! -d "$ROOT/Coll_Models_v2" ]]; then
    printf 'ERROR: submit this file from the DSMC_V2 repository root.\n' >&2
    exit 2
fi
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# The kinematic propensity dominates this stage at production sample counts.
# 64 impact-plane samples cost about 3 percent of the effective sample size in
# weight noise and halve the run time; see reports/energy_kernel_change.md.
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
    hpc/python.sh hpc/estimate_closure_task.py \
    --manifest manifests/closure_baseline.csv \
    --index "$SLURM_ARRAY_TASK_ID" \
    --output results/closure_estimates/baseline \
    --bootstrap 200 --propensity-offsets 64
