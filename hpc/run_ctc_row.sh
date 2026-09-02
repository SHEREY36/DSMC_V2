#!/bin/bash
# Execute and finalize one CSV row from a CTC sweep manifest.
set -euo pipefail

ROW=${1:?CSV manifest row is required}
EXPECTED_TASK=${2:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

FIELD_COUNT=$(awk -F, '{print NF}' <<< "$ROW")
if [[ "$FIELD_COUNT" == 10 ]]; then
    IFS=, read -r TASK_ID STAGE ROLE ALPHA THETA AR SEED NSAMPLES SHARD OUTPUT_DIR <<< "$ROW"
    ENSEMBLE_ID=0
elif [[ "$FIELD_COUNT" == 13 ]]; then
    IFS=, read -r TASK_ID STAGE ROLE ALPHA THETA AR ENSEMBLE_ID ENSEMBLE_MODE \
        CONTROL SEED NSAMPLES SHARD OUTPUT_DIR <<< "$ROW"
else
    printf 'ERROR: expected a 10-field legacy or 13-field closure manifest row, got %s.\n' \
        "$FIELD_COUNT" >&2
    exit 2
fi
# Accept manifests produced before v2.1.1, whose CRLF row ending otherwise
# becomes an invisible carriage return in the output directory name.
OUTPUT_DIR=${OUTPUT_DIR%$'\r'}
if [[ -n "$EXPECTED_TASK" && "$TASK_ID" != "$EXPECTED_TASK" ]]; then
    printf 'ERROR: expected task %s but manifest returned %s.\n' "$EXPECTED_TASK" "$TASK_ID" >&2
    exit 2
fi

printf 'task=%s stage=%s role=%s alpha=%s theta=%s AR=%s ensemble=%s seed=%s samples=%s shard=%s\n' \
    "$TASK_ID" "$STAGE" "$ROLE" "$ALPHA" "$THETA" "$AR" "$ENSEMBLE_ID" \
    "$SEED" "$NSAMPLES" "$SHARD"

if [[ "${DSMC_DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi
if [[ ! -x "$ROOT/HS_CTC_v2/build/SphCyl" ]]; then
    printf 'ERROR: HS_CTC_v2/build/SphCyl is missing. Run the documented build first.\n' >&2
    exit 2
fi
if [[ -f "$ROOT/$OUTPUT_DIR/_SUCCESS" ]]; then
    printf 'Output already complete; skipping %s.\n' "$OUTPUT_DIR"
    exit 0
fi
if [[ -e "$ROOT/$OUTPUT_DIR" ]]; then
    printf 'ERROR: incomplete target already exists; refusing to overwrite %s.\n' \
        "$OUTPUT_DIR" >&2
    printf 'Move it into results/quarantine and resubmit this task.\n' >&2
    exit 3
fi

mkdir -p "$(dirname "$ROOT/$OUTPUT_DIR")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-cores}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT/HS_CTC_v2"
START_SECONDS=$SECONDS
srun ./build/SphCyl \
    "$ALPHA" "$THETA" 1.0 "$AR" "$ROOT/$OUTPUT_DIR" \
    "$SEED" "$NSAMPLES" v2 "$ENSEMBLE_ID"

PYTHONPATH="$ROOT/contracts/python" \
"$ROOT/hpc/python.sh" scripts/finalize_run.py "$ROOT/$OUTPUT_DIR"
ELAPSED_SECONDS=$((SECONDS - START_SECONDS))
"$ROOT/hpc/python.sh" scripts/record_runtime.py "$ROOT/$OUTPUT_DIR" \
    --seconds "$ELAPSED_SECONDS"
