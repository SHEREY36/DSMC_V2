#!/bin/bash -l
# Release the USF pilot only if the rebuilt artifact passes both independent gates.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/usf_preflight_%j.out
#SBATCH --error=logs/usf_preflight_%j.err

set -euo pipefail
HCS=${1:?HCS summary is required}
HOLDOUT=${2:?fresh-CTC validation is required}
ARTIFACT=${3:?candidate artifact is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
hpc/python.sh hpc/require_usf_prerequisites.py \
  --hcs "$HCS" --holdout "$HOLDOUT" --artifact "$ARTIFACT"
