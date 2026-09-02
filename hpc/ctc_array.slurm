#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=ctc_closure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ctc_%A_%a.out
#SBATCH --error=logs/ctc_%A_%a.err

set -euo pipefail
MANIFEST=${1:?manifest path is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
if [[ ! -d "$ROOT/HS_CTC_v2" ]]; then
    printf 'ERROR: submit this file from the DSMC_V2 repository root.\n' >&2
    exit 2
fi
cd "$ROOT"
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
exec bash hpc/run_ctc_row.sh "$ROW" "$SLURM_ARRAY_TASK_ID"
