#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=exc_surface_qa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exc_surface_qa_%j.out
#SBATCH --error=logs/exc_surface_qa_%j.err

set -euo pipefail
MANIFEST=${1:?manifest required}
SUMMARY=${2:?summary required}
OFFLINE=${3:?offline validation required}
EXPECTED_MODE=${4:-}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/summarize_excitation.py --manifest "$MANIFEST" --output "$SUMMARY"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh Coll_Models_v2/scripts/validate_multivariate_corrections.py \
    --manifest "$MANIFEST" --output "$OFFLINE"
ARGS=("$SUMMARY")
[[ -n "$EXPECTED_MODE" ]] && ARGS+=(--expected-mode "$EXPECTED_MODE")
hpc/python.sh hpc/require_excitation_pass.py "${ARGS[@]}"
hpc/python.sh -c \
  'import json,sys; assert json.load(open(sys.argv[1]))["offline_validation_pass"]' \
  "$OFFLINE"
