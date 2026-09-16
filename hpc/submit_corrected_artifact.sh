#!/bin/bash
# Build a correction-enabled candidate only after the 3-D response grid passes.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
GRID=${1:-manifests/artifact_grid.csv}
BASELINES=${2:-results/closure_estimates/artifact_grid}
EXCITATIONS=${3:-results/closure_estimates/excitation_correction-grid}
SUMMARY=${4:-results/closure_estimates/excitation_correction-grid_summary.json}
OUTPUT=${5:-models/microscopic_closure_v2_candidate}
WORK=${6:-results/closure_estimates/artifact_precompute_corrected}
OFFLINE=${7:-results/closure_estimates/excitation_correction-grid_offline_validation.json}

mkdir -p logs "$OUTPUT" "$WORK"
# Recompute both gates with the checked-out code.  This avoids trusting a
# stale summary copied from an earlier summarizer implementation.
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/summarize_excitation.py \
  --manifest manifests/excitation_correction-grid.csv --output "$SUMMARY"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh Coll_Models_v2/scripts/validate_multivariate_corrections.py \
  --manifest manifests/excitation_correction-grid.csv --output "$OFFLINE"
hpc/python.sh -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["offline_validation_pass"], "offline multivariate response validation failed"' "$OFFLINE"
hpc/python.sh hpc/require_excitation_pass.py "$SUMMARY" \
  --expected-mode correction-grid
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/validate_artifact_inputs.py \
  --manifest "$GRID" --estimates "$BASELINES" \
  --output results/closure_estimates/corrected_artifact_baseline_validation.json \
  --require-current-estimates --require-pass

ROWS=$(( $(wc -l < "$GRID") - 1 ))
MAX_ARRAY=$(scontrol show config 2>/dev/null \
  | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
MAX_ARRAY=${MAX_ARRAY:-1000}
TASKS=$ROWS
(( TASKS > MAX_ARRAY )) && TASKS=$MAX_ARRAY
CONCURRENT=$(( ${ARTIFACT_MAX_CORES:-256} / 2 ))
(( CONCURRENT > TASKS )) && CONCURRENT=$TASKS
PRE_RAW=$(sbatch --parsable --array="0-$((TASKS - 1))%$CONCURRENT" \
  hpc/artifact_precompute_stride.slurm \
  "$GRID" "$BASELINES" "$WORK" "$ROWS" "$EXCITATIONS")
PRE_JOB=${PRE_RAW%%;*}
PACK_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PRE_JOB" hpc/aggregate.slurm \
  "$GRID" "$BASELINES" "$OUTPUT" "$WORK" "$EXCITATIONS")
PACK_JOB=${PACK_RAW%%;*}
echo "corrected_precompute_job=$PRE_JOB ($CONCURRENT simultaneous x 2 cores)"
echo "corrected_artifact_job=$PACK_JOB (afterok:$PRE_JOB)"
echo "Candidate only: production artifact is not overwritten."
bash hpc/submit_corrected_hcs_validation.sh \
  "$OUTPUT/closure_v2.npz" "$PACK_JOB"
