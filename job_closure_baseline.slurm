#!/bin/bash -l
# Conventional Negishi submission: sbatch job_closure_baseline.slurm
# Requires manifests/closure_baseline.csv (1248 rows; see README).
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1247%40
#SBATCH --job-name=ctc_baseline
#SBATCH --output=logs/ctc_baseline_%A_%a.out
#SBATCH --error=logs/ctc_baseline_%A_%a.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
if [[ ! -d "$ROOT/HS_CTC_v2" ]]; then
    printf 'ERROR: submit this file from the DSMC_V2 repository root.\n' >&2
    exit 2
fi
cd "$ROOT"
MANIFEST=manifests/closure_baseline.csv
if [[ ! -f "$MANIFEST" ]]; then
    printf 'ERROR: %s is missing; generate it using the README command.\n' \
        "$MANIFEST" >&2
    exit 2
fi
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
if [[ -z "$ROW" ]]; then
    printf 'ERROR: manifest has no row for array index %s.\n' \
        "$SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
exec bash hpc/run_ctc_row.sh "$ROW" "$SLURM_ARRAY_TASK_ID"
