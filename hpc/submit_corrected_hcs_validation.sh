#!/bin/bash
# Validate correction-enabled dynamics without overwriting the baseline HCS gate.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
ARTIFACT=${1:-models/microscopic_closure_v2_candidate/closure_v2.npz}
DEPENDENCY=${2:-}
MANIFEST=${3:-manifests/hcs_corrected_validation.csv}
RESULTS=${4:-results/hcs_corrected_validation}

mkdir -p logs "$RESULTS"
hpc/python.sh DSMC_0D_v2/scripts/make_hcs_validation_manifest.py \
  --output "$MANIFEST" --results "$RESULTS"
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
CONCURRENT=$ROWS
(( CONCURRENT > ${HCS_MAX_CORES:-256} )) && CONCURRENT=${HCS_MAX_CORES:-256}
SBATCH_ARGS=(--parsable --array="0-$((ROWS - 1))%$CONCURRENT")
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--kill-on-invalid-dep=yes --dependency="afterok:$DEPENDENCY")
fi
RAW=$(sbatch "${SBATCH_ARGS[@]}" hpc/hcs_validation_array.slurm \
  "$MANIFEST" "$ARTIFACT" true)
JOB=${RAW%%;*}
PLOT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$JOB" hpc/plot_hcs_validation.slurm "$MANIFEST" \
  "$RESULTS/hcs_theta_attraction.png" "$RESULTS/summary.json")
PLOT=${PLOT_RAW%%;*}
echo "corrected_hcs_job=$JOB ($ROWS one-core tasks; concurrency=$CONCURRENT)"
echo "corrected_hcs_plot_job=$PLOT (afterok:$JOB)"
echo "Inspect $RESULTS/summary.json before any direct-CTC or production-grid run."
