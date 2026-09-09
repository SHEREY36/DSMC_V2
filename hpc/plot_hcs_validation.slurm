#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=hcs_plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/hcs_plot_%j.out
#SBATCH --error=logs/hcs_plot_%j.err

set -euo pipefail
MANIFEST=${1:-manifests/hcs_validation.csv}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dsmc-v2-matplotlib-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
hpc/python.sh DSMC_0D_v2/scripts/plot_hcs_validation.py \
  --manifest "$MANIFEST" \
  --figure results/hcs_validation/hcs_theta_attraction.png \
  --summary results/hcs_validation/summary.json
