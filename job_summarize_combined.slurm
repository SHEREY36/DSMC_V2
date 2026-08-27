#!/bin/bash -l
# Submit after job_estimate_combined.slurm completes successfully.
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=summarize_combined
#SBATCH --output=summarize_combined_%j.out
#SBATCH --error=summarize_combined_%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
exec hpc/python.sh hpc/summarize_estimates.py \
    --input coefficients/combined_nodes --output coefficients/combined --expected 870
