#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_promote
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/usf_promote_%j.out
#SBATCH --error=logs/usf_promote_%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
hpc/python.sh DSMC_0D_v2/scripts/promote_usf_validated_artifact.py \
  --artifact "${1:?artifact required}" --hcs-summary "${2:?HCS summary required}" \
  --usf-summary "${3:?USF summary required}" \
  --reference-provenance DSMC_0D_v2/reference/usf_reference_provenance_v1.json \
  --output-dir "${4:?output directory required}"
