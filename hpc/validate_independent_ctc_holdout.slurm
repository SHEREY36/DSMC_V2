#!/bin/bash -l
# Compare fresh CTC holdout responses against the frozen candidate artifact.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=ctc_holdout_qa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ctc_holdout_qa_%j.out
#SBATCH --error=logs/ctc_holdout_qa_%j.err

set -euo pipefail
MANIFEST=${1:?excitation manifest is required}
BASELINE=${2:?fresh baseline estimate is required}
ARTIFACT=${3:?candidate artifact is required}
OUTPUT=${4:?validation JSON is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1

hpc/python.sh hpc/validate_independent_ctc_holdout.py \
  --manifest "$MANIFEST" \
  --baseline-estimate "$BASELINE" \
  --artifact "$ARTIFACT" \
  --output "$OUTPUT"
