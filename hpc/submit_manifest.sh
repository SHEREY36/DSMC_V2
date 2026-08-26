#!/bin/bash
set -euo pipefail
MANIFEST=${1:?usage: submit_manifest.sh MANIFEST.csv}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p logs manifests/split
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk '/MaxArraySize/ {print $3; exit}')
MAX_ARRAY=${MAX_ARRAY:-1000}
mapfile -t PARTS < <(python3 hpc/split_manifest.py "$MANIFEST" --max-rows "$MAX_ARRAY" --output-dir manifests/split)
for PART in "${PARTS[@]}"; do
    ROWS=$(( $(wc -l < "$PART") - 1 ))
    if (( ROWS > 0 )); then
        sbatch --array="0-$((ROWS - 1))%50" hpc/ctc_array.slurm "$PART"
    fi
done
