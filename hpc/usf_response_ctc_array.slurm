#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=ctc_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ctc_replay_%A_%a.out
#SBATCH --error=logs/ctc_replay_%A_%a.err
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd "$ROOT"
hpc/run_usf_response_ctc_row.sh "${1:?manifest required}" "$SLURM_ARRAY_TASK_ID"
