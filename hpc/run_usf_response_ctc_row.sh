#!/bin/bash
# Run one exact CTC equilibrium-anchor or DSMC-replay task.
set -euo pipefail
MANIFEST=${1:?manifest path is required}
INDEX=${2:?task index is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"

resolve_path() {
  if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$ROOT" "$1"; fi
}

ROW=$(awk -F, -v target="$INDEX" 'NR==target+2 {print; exit}' "$MANIFEST")
[[ -n "$ROW" ]] || { echo "manifest row $INDEX is missing" >&2; exit 2; }
IFS=, read -r TASK ROLE ALPHA THETA AR SEED NSAMPLES ENSEMBLE SOURCE REPLAY \
  INITIAL_BRANCH WINDOW_INDEX ARTIFACT ARTIFACT_SHA OUTPUT_DIR <<< "${ROW%$'\r'}"
[[ "$TASK" == "$INDEX" ]] || { echo "manifest task/index mismatch" >&2; exit 2; }
if [[ -f "$OUTPUT_DIR/_SUCCESS" ]]; then
  echo "CTC task already complete: $OUTPUT_DIR"
  exit 0
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "incomplete CTC output exists; move it to quarantine: $OUTPUT_DIR" >&2
  exit 3
fi
[[ -x HS_CTC_v2/build/SphCyl ]] || { echo "CTC executable is missing" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT_DIR")"
# Run directly inside the batch allocation. Negishi may grant an extra billed
# CPU to satisfy a whole-job memory request, leaving SLURM_CPUS_PER_TASK and
# SLURM_TRES_PER_TASK inconsistent; a nested srun then aborts before launch.
# The CTC design and array accounting deliberately use four OpenMP threads.
export OMP_NUM_THREADS=${CTC_OMP_THREADS:-4}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-cores}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

START=$SECONDS
cd HS_CTC_v2
if [[ "$ROLE" == usf_replay ]]; then
  SOURCE_PATH=$(resolve_path "$SOURCE")
  REPLAY_PATH=$(resolve_path "$REPLAY")
  OUTPUT_PATH=$(resolve_path "$OUTPUT_DIR")
  [[ -f "$SOURCE_PATH" && -f "$REPLAY_PATH" ]] || {
    echo "replay source files are missing" >&2; exit 2; }
  ./build/SphCyl "$ALPHA" "$THETA" 1.0 "$AR" "$OUTPUT_PATH" \
    "$SEED" "$NSAMPLES" v2 "$ENSEMBLE" "$REPLAY_PATH"
  cd "$ROOT"
  hpc/python.sh hpc/attach_replay_provenance.py "$OUTPUT_PATH" "$SOURCE_PATH"
else
  echo "unsupported CTC response role: $ROLE" >&2
  exit 2
fi
PYTHONPATH="$ROOT/contracts/python" hpc/python.sh \
  HS_CTC_v2/scripts/finalize_run.py "$OUTPUT_PATH"
hpc/python.sh HS_CTC_v2/scripts/record_runtime.py "$OUTPUT_PATH" \
  --seconds "$((SECONDS - START))"
