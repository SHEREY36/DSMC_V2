#!/bin/bash
# Submit the compact two-initial-condition HCS gate after artifact construction.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MANIFEST=${1:-manifests/hcs_validation.csv}
ARTIFACT=${2:-models/microscopic_closure_v2/closure_v2.npz}
DEPENDENCY=${3:-}

mkdir -p logs results/hcs_validation
hpc/python.sh DSMC_0D_v2/scripts/make_hcs_validation_manifest.py --output "$MANIFEST"
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
MAX_CORES=${HCS_MAX_CORES:-256}
CONCURRENT=$ROWS
(( CONCURRENT > MAX_CORES )) && CONCURRENT=$MAX_CORES
SBATCH_ARGS=(--parsable --array="0-$((ROWS - 1))%$CONCURRENT")
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--kill-on-invalid-dep=yes --dependency="afterok:$DEPENDENCY")
fi
JOB_RAW=$(sbatch "${SBATCH_ARGS[@]}" hpc/hcs_validation_array.slurm "$MANIFEST" "$ARTIFACT")
JOB=${JOB_RAW%%;*}
PLOT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$JOB" hpc/plot_hcs_validation.slurm "$MANIFEST")
PLOT_JOB=${PLOT_RAW%%;*}
echo "hcs_job=$JOB ($ROWS independent one-core tasks; concurrency=$CONCURRENT)"
echo "hcs_plot_job=$PLOT_JOB (afterok:$JOB)"
echo "After completion inspect results/hcs_validation/{summary.json,hcs_theta_attraction.png}"
