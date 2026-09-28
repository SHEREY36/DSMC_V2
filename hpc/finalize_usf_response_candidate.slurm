#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=resp_promote
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/resp_promote_%j.out
#SBATCH --error=logs/resp_promote_%j.err
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd "$ROOT"
hpc/python.sh DSMC_0D_v2/scripts/finalize_usf_response_candidate.py \
  --artifact "${1:?artifact}" --direct-report "${2:?direct report}" \
  --hcs-summary "${3:?HCS summary}" --usf-summary "${4:?USF summary}" \
  --output "${5:?promotion decision}"
