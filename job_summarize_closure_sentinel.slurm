#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -J closure_summary_sentinel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH -o logs/closure_summary_sentinel_%j.out
#SBATCH -e logs/closure_summary_sentinel_%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
if [[ ! -d "$ROOT/Coll_Models_v2" ]]; then
    printf 'ERROR: submit this file from the DSMC_V2 repository root.\n' >&2
    exit 2
fi
cd "$ROOT"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
    hpc/python.sh hpc/summarize_sentinel.py \
    --runs-root results/ctc_closure/sentinel \
    --estimates results/closure_estimates/sentinel \
    --manifest manifests/closure_sentinel.csv \
    --output results/closure_estimates/sentinel_report.json
