#!/bin/bash -l
# Submit after job_estimate_pilot.slurm completes successfully.
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=summarize_pilot
#SBATCH --output=summarize_pilot_%j.out
#SBATCH --error=summarize_pilot_%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
exec hpc/python.sh hpc/summarize_estimates.py \
    --input coefficients/pilot_nodes --output coefficients/pilot --expected 870
