#!/bin/bash -l
# Submit only after combined QA requests more samples.
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-869%50
#SBATCH --job-name=continuation_v2
#SBATCH --output=continuation_v2_%A_%a.out
#SBATCH --error=continuation_v2_%A_%a.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
if [[ ! -d "$ROOT/HS_CTC_v2" ]]; then
    printf 'ERROR: submit this file from the DSMC_V2 repository root.\n' >&2
    exit 2
fi
cd "$ROOT"
exec hpc/run_continuation_task.sh
