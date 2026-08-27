#!/bin/bash
# Execute one QA-selected continuation row. Unused fixed-array slots exit cleanly.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?this runner requires a Slurm array index}
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
SHARD=${DSMC_SHARD:-2}
SAMPLES=${DSMC_SAMPLES:-100000}
QA_SUMMARY=${DSMC_QA_SUMMARY:-"$ROOT/coefficients/combined/qa_summary.csv"}
if [[ ! -f "$QA_SUMMARY" ]]; then
    printf 'ERROR: combined QA summary not found: %s\n' "$QA_SUMMARY" >&2
    exit 2
fi

TASK_TMP=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
TASK_MANIFEST="$TASK_TMP/dsmc_v2_continuation_${ARRAY_JOB_ID}_${TASK_INDEX}.csv"
"$ROOT/hpc/python.sh" "$ROOT/hpc/make_manifest.py" \
    --stage continuation --samples "$SAMPLES" --shard "$SHARD" \
    --qa-summary "$QA_SUMMARY" --output "$TASK_MANIFEST"
if [[ "$TASK_INDEX" == 0 ]]; then
    mkdir -p "$ROOT/manifests"
    cp "$TASK_MANIFEST" "$ROOT/manifests/continuation_$(printf '%02d' "$SHARD").csv"
fi
ROW=$(sed -n "$((TASK_INDEX + 2))p" "$TASK_MANIFEST")
if [[ -z "$ROW" ]]; then
    printf 'No continuation case assigned to array slot %s; exiting.\n' "$TASK_INDEX"
    exit 0
fi
exec "$ROOT/hpc/run_ctc_row.sh" "$ROW" "$TASK_INDEX"
