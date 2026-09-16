#!/bin/bash -l
# Validate repaired estimates against the already deep-validated canonical raw grid.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=validate_artifact_estimates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validate_artifact_estimates_%j.out
#SBATCH --error=logs/validate_artifact_estimates_%j.err

set -euo pipefail
MANIFEST=${1:-manifests/artifact_grid.csv}
ESTIMATES=${2:-results/closure_estimates/artifact_grid}
REPORT=${3:-results/closure_estimates/artifact_grid_validation.json}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/validate_artifact_inputs.py \
  --manifest "$MANIFEST" --estimates "$ESTIMATES" --output "$REPORT" \
  --require-current-estimates --require-pass
