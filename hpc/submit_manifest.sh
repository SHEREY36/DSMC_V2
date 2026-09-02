#!/bin/bash
set -euo pipefail
MANIFEST=${1:?usage: submit_manifest.sh MANIFEST.csv}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p logs manifests/split
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk '/MaxArraySize/ {print $3; exit}')
MAX_ARRAY=${MAX_ARRAY:-1000}
MAX_CONCURRENT=${CTC_MAX_CONCURRENT:-12}
if (( MAX_CONCURRENT < 1 )); then
    printf 'ERROR: CTC_MAX_CONCURRENT must be positive.\n' >&2
    exit 2
fi
mapfile -t PARTS < <(hpc/python.sh hpc/split_manifest.py "$MANIFEST" --max-rows "$MAX_ARRAY" --output-dir manifests/split)
for PART in "${PARTS[@]}"; do
    ROWS=$(( $(wc -l < "$PART") - 1 ))
    if (( ROWS > 0 )); then
        sbatch --array="0-$((ROWS - 1))%$MAX_CONCURRENT" hpc/ctc_array.slurm "$PART"
    fi
done
