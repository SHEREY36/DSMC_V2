#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=usf_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/usf_analyze_%j.out
#SBATCH --error=logs/usf_analyze_%j.err

set -euo pipefail
MANIFEST=${1:?USF manifest is required}
SUMMARY=${2:?summary path is required}
FIGURE=${3:?figure path is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dsmc-v2-usf-matplotlib-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
hpc/python.sh DSMC_0D_v2/scripts/analyze_usf_validation.py \
  --manifest "$MANIFEST" --summary "$SUMMARY" --figure "$FIGURE"
