#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=hcs_ng_qa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hcs_ng_qa_%j.out
#SBATCH --error=logs/hcs_ng_qa_%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dsmc-v2-hcs-ng-matplotlib-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
PYTHONPATH="$ROOT/contracts/python:$ROOT/DSMC_0D_v2/src" \
  hpc/python.sh DSMC_0D_v2/scripts/analyze_hcs_ng_campaign.py \
    --manifest "${1:?manifest required}" --output "${2:?summary required}" \
    --figure "${3:?figure required}"
