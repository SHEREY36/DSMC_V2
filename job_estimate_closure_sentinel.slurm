#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -J closure_fit_sentinel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 06:00:00
#SBATCH --array=0-35
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
# At 200,000 hits the kinematic propensity and the bootstrap dominate this
# stage. 64 impact-plane samples cost about 3 per cent of the effective sample
# size in weight noise.
#
# Sizing, measured rather than guessed:
#   * peak RSS is 0.92 GB, so --mem=4G is generous. The old 16G forced Slurm to
#     charge ~8 cores per single-threaded task and throttled the array to 108
#     of 256 CPUs.
#   * the %12 array throttle was the other half of that: 36 tasks at -c 2 is
#     72 cores, so the whole sentinel runs in one wave.
#   * the bridge fits at 128-point quadrature (lambda3 agrees with 256 to seven
#     significant figures). That is a 4x speedup and is what puts 200 bootstrap
#     replicates inside the wall: ~133 s point fit plus 200 x 26 s is about 2 h.
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
    hpc/python.sh hpc/estimate_closure_task.py \
    --manifest manifests/closure_sentinel.csv \
    --index "$SLURM_ARRAY_TASK_ID" \
    --output results/closure_estimates/sentinel \
    --bootstrap 200 --propensity-offsets 64
