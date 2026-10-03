#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=hcs_cadence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/hcs_cadence_%j.out
#SBATCH --error=logs/hcs_cadence_%j.err

set -euo pipefail
REFERENCE=${1:?reference manifest is required}
OPTIMIZED=${2:?optimized manifest is required}
OUTPUT=${3:?output JSON is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
hpc/python.sh DSMC_0D_v2/scripts/compare_hcs_update_cadence.py \
  --reference-manifest "$REFERENCE" --optimized-manifest "$OPTIMIZED" \
  --output "$OUTPUT"
