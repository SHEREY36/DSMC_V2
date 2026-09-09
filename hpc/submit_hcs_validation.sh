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
SBATCH_ARGS=(--parsable --array="0-$((ROWS - 1))%20")
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--kill-on-invalid-dep=yes --dependency="afterok:$DEPENDENCY")
fi
JOB_RAW=$(sbatch "${SBATCH_ARGS[@]}" hpc/hcs_validation_array.slurm "$MANIFEST" "$ARTIFACT")
JOB=${JOB_RAW%%;*}
echo "hcs_job=$JOB"
echo "After completion: PYTHONPATH=DSMC_0D_v2/src hpc/python.sh DSMC_0D_v2/scripts/plot_hcs_validation.py"
