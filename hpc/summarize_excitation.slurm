#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=excitation_qa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/excitation_qa_%j.out
#SBATCH --error=logs/excitation_qa_%j.err

set -euo pipefail
MANIFEST=${1:?excitation manifest is required}
OUTPUT=${2:?excitation summary path is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/summarize_excitation.py --manifest "$MANIFEST" --output "$OUTPUT"
