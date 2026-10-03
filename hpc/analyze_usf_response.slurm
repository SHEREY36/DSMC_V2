#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_resp_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/usf_resp_gate_%j.out
#SBATCH --error=logs/usf_resp_gate_%j.err
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
LOCAL=${1:?local coefficient output}; DIRECT=${2:?direct estimates}
FINAL=${3:?final coefficient output}; REPORT=${4:?report output}
BASE=${5:?base artifact}
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh Coll_Models_v2/scripts/compile_correction_surface.py \
  --node-estimates results/closure_estimates/artifact_grid \
  results/closure_estimates/excitation_correction-grid \
  results/closure_estimates/excitation_usf_extension_v1 \
  results/closure_estimates/excitation_full_candidate_v1_support_refinement \
  results/closure_estimates/excitation_full_candidate_v1_missing \
  --output "$LOCAL" --response-order 2 --release-policy quadratic-evidence-v1
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src" \
  hpc/python.sh Coll_Models_v2/scripts/analyze_usf_direct_response.py \
  --base-artifact "$BASE" --coefficient-rows "$LOCAL" \
  --direct-estimates "$DIRECT" --output-coefficients "$FINAL" \
  --output-report "$REPORT"
