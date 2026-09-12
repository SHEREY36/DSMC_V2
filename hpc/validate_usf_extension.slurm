#!/bin/bash -l
# Fail-closed response and distribution validation for the USF hull extension.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_ext_qa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/usf_ext_qa_%j.out
#SBATCH --error=logs/usf_ext_qa_%j.err

set -euo pipefail
MANIFEST=${1:?excitation manifest is required}
SUMMARY=${2:?summary JSON is required}
OFFLINE=${3:?offline validation JSON is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/summarize_excitation.py \
    --manifest "$MANIFEST" --output "$SUMMARY"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh Coll_Models_v2/scripts/validate_multivariate_corrections.py \
    --manifest "$MANIFEST" --output "$OFFLINE"
hpc/python.sh hpc/require_excitation_pass.py "$SUMMARY" \
  --expected-mode usf-extension
hpc/python.sh -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["offline_validation_pass"], "offline validation failed"' \
  "$OFFLINE"
