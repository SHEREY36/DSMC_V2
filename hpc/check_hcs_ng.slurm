#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=hcs_ng_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/hcs_ng_preflight_%j.out
#SBATCH --error=logs/hcs_ng_preflight_%j.err

set -euo pipefail
MANIFEST=${1:?manifest required}
ARTIFACT=${2:?artifact required}
HCS_SUMMARY=${3:?full-domain HCS summary required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
PYTHONPATH="$ROOT/DSMC_0D_v2/src" hpc/python.sh \
  hpc/check_hcs_ng_prerequisites.py --manifest "$MANIFEST" \
  --artifact "$ARTIFACT" --hcs-summary "$HCS_SUMMARY"
