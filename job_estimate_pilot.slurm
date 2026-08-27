#!/bin/bash -l
# Decode and estimate the completed 20k pilot, one parameter node per task.
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-869%50
#SBATCH --job-name=estimate_pilot
#SBATCH --output=estimate_pilot_%A_%a.out
#SBATCH --error=estimate_pilot_%A_%a.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec hpc/python.sh hpc/estimate_task.py \
    --index "$SLURM_ARRAY_TASK_ID" --scope pilot \
    --runs-root results/ctc --output coefficients/pilot_nodes --bootstrap 2000 \
    --gamma-max-table Coll_Models_v2/models/legacy_bl/gamma_max_table.json \
    --one-hit-table Coll_Models_v2/models/legacy_bl/one_hit_table.json
