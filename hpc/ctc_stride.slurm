#!/bin/bash -l
# Generate a whole CTC manifest with one plain sbatch call.
#
#   sbatch --array=0-11 hpc/ctc_stride.slurm manifests/closure_baseline.csv
#
# Each array task walks the manifest with a stride equal to the array width, so
# the array size is independent of the manifest length and never runs into the
# cluster's MaxArraySize. Rows already carrying _SUCCESS are skipped, so the
# same command is also the restart command.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=ctc_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ctc_gen_%A_%a.out
#SBATCH --error=logs/ctc_gen_%A_%a.err

set -uo pipefail          # deliberately not -e: one bad row must not kill the rest
MANIFEST=${1:?usage: sbatch --array=0-N hpc/ctc_stride.slurm MANIFEST.csv}
cd "${SLURM_SUBMIT_DIR:?}"

STRIDE=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
START=${SLURM_ARRAY_TASK_ID:-0}
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
printf 'task %s: manifest %s has %s rows, stride %s\n' "$START" "$MANIFEST" "$ROWS" "$STRIDE"

FAILED=0
for (( i = START; i < ROWS; i += STRIDE )); do
    ROW=$(sed -n "$(( i + 2 ))p" "$MANIFEST")
    [[ -n "$ROW" ]] || continue
    printf '=== row %s\n' "$i"
    if ! bash hpc/run_ctc_row.sh "$ROW" "$i"; then
        printf 'row %s FAILED\n' "$i" >&2
        FAILED=$(( FAILED + 1 ))
    fi
done
printf 'task %s finished with %s failed rows\n' "$START" "$FAILED"
exit $(( FAILED > 0 ))
