#!/bin/bash -l
# Fit one part of a split closure manifest. Submitted by submit_closure_fits.sh,
# which handles the cluster's MaxArraySize for you.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=closure_fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# The fit is single threaded and peaks near 0.7 GB on a 200,000-hit shard, 1.4
# on the heaviest. Requesting more does not make it faster: Negishi allocates
# about 2 GB per core, so an oversized --mem silently reserves cores the task
# never uses and throttles the array to a fraction of the account's CPUs.
#SBATCH --time=12:00:00
#SBATCH --output=logs/closure_fit_%A_%a.out
#SBATCH --error=logs/closure_fit_%A_%a.err

set -euo pipefail
MANIFEST=${1:?manifest path is required}
OUTPUT=${2:?output directory is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
    hpc/python.sh hpc/estimate_closure_task.py \
    --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --output "$OUTPUT" \
    --bootstrap "${CLOSURE_BOOTSTRAP:-200}" \
    --propensity-offsets "${CLOSURE_OFFSETS:-64}"
