#!/bin/bash -l
# Estimate pilot + production + any successful continuation shards.
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-869%50
#SBATCH --job-name=estimate_combined
#SBATCH --output=estimate_combined_%A_%a.out
#SBATCH --error=estimate_combined_%A_%a.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec hpc/python.sh hpc/estimate_task.py \
    --index "$SLURM_ARRAY_TASK_ID" --scope combined \
    --runs-root results/ctc --output coefficients/combined_nodes --bootstrap 2000 \
    --gamma-max-table Coll_Models_v2/models/legacy_bl/gamma_max_table.json \
    --one-hit-table Coll_Models_v2/models/legacy_bl/one_hit_table.json
