#!/bin/bash
# Split a closure manifest to the cluster's MaxArraySize and submit the fits.
#   bash hpc/submit_closure_fits.sh manifests/closure_baseline.csv \
#        results/closure_estimates/baseline
set -euo pipefail
MANIFEST=${1:?usage: submit_closure_fits.sh MANIFEST.csv OUTPUT_DIR}
OUTPUT=${2:?usage: submit_closure_fits.sh MANIFEST.csv OUTPUT_DIR}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p logs manifests/split "$OUTPUT"
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk '/MaxArraySize/ {print $3; exit}')
MAX_ARRAY=${MAX_ARRAY:-1000}
# MaxArraySize is the exclusive upper bound on the index, so the largest usable
# array is 0..MAX_ARRAY-1.
(( MAX_ARRAY > 1 )) && MAX_ARRAY=$(( MAX_ARRAY - 1 ))
# One core per task, so this is the number of CPUs the array will hold.
# Size it to the account's CPU allocation (slist) rather than a round number.
MAX_CONCURRENT=${CLOSURE_MAX_CONCURRENT:-256}
mapfile -t PARTS < <(hpc/python.sh hpc/split_manifest.py "$MANIFEST" \
    --max-rows "$MAX_ARRAY" --output-dir manifests/split)
for PART in "${PARTS[@]}"; do
    ROWS=$(( $(wc -l < "$PART") - 1 ))
    if (( ROWS > 0 )); then
        sbatch --array="0-$((ROWS - 1))%$MAX_CONCURRENT" \
            hpc/closure_fit_array.slurm "$PART" "$OUTPUT"
    fi
done
