#!/bin/bash
# Submit a reweighted pilot only after baseline HCS passes.  No fresh CTC runs.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-hcs-pilot}
HCS_SUMMARY=${2:-results/hcs_validation/summary.json}
MANIFEST=${3:-manifests/excitation_${MODE}.csv}
RESULTS=${4:-results/closure_estimates/excitation_${MODE}}
SUMMARY=${5:-results/closure_estimates/excitation_${MODE}_summary.json}

mkdir -p logs "$RESULTS" "$(dirname "$SUMMARY")"
hpc/python.sh hpc/require_hcs_pass.py "$HCS_SUMMARY"
case "$MODE" in
  hcs-pilot) ;;
  full-pilot)
    hpc/python.sh hpc/require_excitation_pass.py \
      results/closure_estimates/excitation_hcs-pilot_summary.json \
      --expected-mode hcs-pilot
    ;;
  correction-grid)
    hpc/python.sh hpc/require_excitation_pass.py \
      results/closure_estimates/excitation_full-pilot_summary.json \
      --expected-mode full-pilot
    ;;
  production-grid)
    hpc/python.sh hpc/require_excitation_pass.py \
      results/closure_estimates/excitation_correction-grid_summary.json \
      --expected-mode correction-grid
    ;;
  *) echo "unknown excitation mode: $MODE" >&2; exit 2 ;;
esac
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/make_excitation_manifest.py \
    --mode "$MODE" --output "$MANIFEST" --results "$RESULTS"

ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
MAX_ARRAY=$(scontrol show config 2>/dev/null \
  | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
MAX_ARRAY=${MAX_ARRAY:-1000}
ARRAY_TASKS=$ROWS
(( ARRAY_TASKS > MAX_ARRAY )) && ARRAY_TASKS=$MAX_ARRAY
CONCURRENT=${EXCITATION_MAX_CORES:-256}
(( CONCURRENT > ARRAY_TASKS )) && CONCURRENT=$ARRAY_TASKS
if (( ROWS < 1 || ARRAY_TASKS < 1 || CONCURRENT < 1 )); then
  echo "excitation manifest or Slurm array capacity is empty" >&2
  exit 2
fi

FIT_RAW=$(sbatch --parsable --array="0-$((ARRAY_TASKS - 1))%$CONCURRENT" \
  hpc/excitation_fit_stride.slurm "$MANIFEST")
FIT_JOB=${FIT_RAW%%;*}
QA_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$FIT_JOB" hpc/summarize_excitation.slurm "$MANIFEST" "$SUMMARY")
QA_JOB=${QA_RAW%%;*}
echo "excitation_fit_job=$FIT_JOB ($ROWS virtual ensembles; concurrency=$CONCURRENT)"
echo "excitation_qa_job=$QA_JOB (afterok:$FIT_JOB)"
echo "No CTC trajectories and no artifact rebuild were submitted."
echo "After completion inspect $SUMMARY"
