#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_stage_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/usf_stage_gate_%j.out
#SBATCH --error=logs/usf_stage_gate_%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
hpc/python.sh hpc/check_usf_study_stage.py \
  --summary "${1:?summary required}" --expected-mode "${2:?mode required}"
