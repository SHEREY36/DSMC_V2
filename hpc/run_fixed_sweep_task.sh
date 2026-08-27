#!/bin/bash
# Map one fixed 870-node pilot/production array index to a CTC run.
set -euo pipefail

STAGE=${1:?pilot or production is required}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case "$STAGE" in
    pilot) SAMPLES=20000; SHARD=0 ;;
    production) SAMPLES=80000; SHARD=1 ;;
    *) printf 'ERROR: unsupported fixed stage %s.\n' "$STAGE" >&2; exit 2 ;;
esac

TASK_INDEX=${SLURM_ARRAY_TASK_ID:?this runner requires a Slurm array index}
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
TASK_TMP=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
TASK_MANIFEST="$TASK_TMP/dsmc_v2_${STAGE}_${ARRAY_JOB_ID}_${TASK_INDEX}.csv"

"$ROOT/hpc/python.sh" "$ROOT/hpc/make_manifest.py" \
    --stage "$STAGE" --samples "$SAMPLES" --shard "$SHARD" \
    --output "$TASK_MANIFEST"
if [[ "$TASK_INDEX" == 0 ]]; then
    mkdir -p "$ROOT/manifests"
    cp "$TASK_MANIFEST" "$ROOT/manifests/${STAGE}.csv"
fi
ROW=$(sed -n "$((TASK_INDEX + 2))p" "$TASK_MANIFEST")
if [[ -z "$ROW" ]]; then
    printf 'ERROR: array index %s has no %s manifest row.\n' "$TASK_INDEX" "$STAGE" >&2
    exit 2
fi
exec "$ROOT/hpc/run_ctc_row.sh" "$ROW" "$TASK_INDEX"
