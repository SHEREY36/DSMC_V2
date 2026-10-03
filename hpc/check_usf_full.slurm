#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_full_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/usf_full_preflight_%j.out
#SBATCH --error=logs/usf_full_preflight_%j.err

set -euo pipefail
HCS=${1:?HCS summary required}
HOLDOUT=${2:?holdout validation required}
ARTIFACT=${3:?artifact required}
PILOT=${4:?USF pilot summary required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
hpc/python.sh hpc/require_usf_prerequisites.py --hcs "$HCS" \
  --holdout "$HOLDOUT" --artifact "$ARTIFACT" --pilot-summary "$PILOT" \
  --require-full-domain
