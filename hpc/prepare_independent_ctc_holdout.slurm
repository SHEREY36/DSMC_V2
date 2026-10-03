#!/bin/bash -l
# Build the 36 boundary-amplitude holdout rows after the fresh baseline fit.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=ctc_holdout_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/ctc_holdout_prep_%j.out
#SBATCH --error=logs/ctc_holdout_prep_%j.err

set -euo pipefail
FIT_MANIFEST=${1:?fit manifest is required}
ESTIMATES=${2:?fresh baseline-estimate directory is required}
MANIFEST=${3:?excitation manifest is required}
RESULTS=${4:?excitation result directory is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"

mkdir -p "$RESULTS"
hpc/python.sh hpc/make_excitation_manifest.py \
  --mode independent-holdout \
  --grid "$FIT_MANIFEST" \
  --estimates "$ESTIMATES" \
  --output "$MANIFEST" \
  --results "$RESULTS"

ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
if (( ROWS != 36 )); then
  echo "independent holdout must contain exactly 36 rows, got $ROWS" >&2
  exit 2
fi
