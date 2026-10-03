#!/bin/bash
# Submit a fail-fast paired pilot or the gated corrected-only full USF grid.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-pilot}
ARTIFACT=${2:-models/microscopic_closure_v2_usf_candidate/closure_v2.npz}
TAG=${3:-${MODE}_v1}
DEPENDENCY=${4:-}
HCS_SUMMARY=${HCS_SUMMARY:-results/hcs_corrected_validation_usf_candidate_v1/summary.json}
HOLDOUT_SUMMARY=${HOLDOUT_SUMMARY:-results/closure_estimates/independent_ctc_holdout_usf_candidate_v1_validation.json}
PILOT_SUMMARY=${PILOT_SUMMARY:-results/usf_validation_pilot_usf_candidate_v1/summary.json}

if [[ "$MODE" != "pilot" && "$MODE" != "full" ]]; then
  echo "mode must be pilot or full" >&2
  exit 2
fi
RESULTS="results/usf_validation_${TAG}"
MANIFEST="manifests/usf_validation_${TAG}.csv"
if [[ -e "$RESULTS/summary.json" ]] || compgen -G "$RESULTS/*.txt" >/dev/null; then
  echo "refusing to mix a new campaign with existing outputs in $RESULTS" >&2
  echo "choose a new TAG, for example: bash $0 $MODE $ARTIFACT ${TAG}_rerun" >&2
  exit 2
fi

PREREQUISITES=(--hcs "$HCS_SUMMARY" --holdout "$HOLDOUT_SUMMARY" \
  --artifact "$ARTIFACT")
if [[ "$MODE" == "full" ]]; then
  PREREQUISITES+=(--pilot-summary "$PILOT_SUMMARY")
fi
hpc/python.sh hpc/require_usf_prerequisites.py "${PREREQUISITES[@]}"
mkdir -p logs "$RESULTS"
hpc/python.sh DSMC_0D_v2/scripts/make_usf_validation_manifest.py \
  --mode "$MODE" --output "$MANIFEST" --results "$RESULTS"
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
CONCURRENT=$ROWS
(( CONCURRENT > ${USF_MAX_CORES:-256} )) && CONCURRENT=${USF_MAX_CORES:-256}
SBATCH_ARGS=(--parsable --array="0-$((ROWS - 1))%$CONCURRENT")
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--kill-on-invalid-dep=yes --dependency="afterok:$DEPENDENCY")
fi
RAW=$(sbatch "${SBATCH_ARGS[@]}" hpc/usf_validation_array.slurm \
  "$MANIFEST" "$ARTIFACT")
JOB=${RAW%%;*}
ANALYZE_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$JOB" hpc/analyze_usf_validation.slurm \
  "$MANIFEST" "$RESULTS/summary.json" "$RESULTS/usf_validation.png")
ANALYZE=${ANALYZE_RAW%%;*}
echo "usf_${MODE}_job=$JOB ($ROWS one-core tasks; concurrency=$CONCURRENT)"
echo "usf_${MODE}_analysis_job=$ANALYZE (afterok:$JOB)"
echo "Inspect $RESULTS/summary.json before proceeding."
