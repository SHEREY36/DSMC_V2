#!/bin/bash -l
# Deep binary-contract and current-estimate gate for the canonical artifact grid.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=validate_artifact_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/validate_artifact_grid_%j.out
#SBATCH --error=logs/validate_artifact_grid_%j.err

set -euo pipefail
MANIFEST=${1:-manifests/artifact_grid.csv}
ESTIMATES=${2:-results/closure_estimates/artifact_grid}
REPORT=${3:-results/closure_estimates/artifact_grid_validation.json}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/validate_artifact_inputs.py \
  --manifest "$MANIFEST" --estimates "$ESTIMATES" --output "$REPORT" \
  --deep --require-current-estimates --require-pass
