#!/bin/bash -l
# Fit a whole closure manifest with one plain sbatch call.
#
#   sbatch --array=0-255 hpc/closure_fit_stride.slurm \
#          manifests/closure_baseline.csv results/closure_estimates/baseline
#
# Same stride scheme as ctc_stride.slurm. The fit is single threaded and peaks
# near 1.4 GB, so one core and 2G lets an array of 256 hold 256 CPUs instead of
# reserving nine cores per task and idling eight of them.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=closure_fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/closure_fit_%A_%a.out
#SBATCH --error=logs/closure_fit_%A_%a.err

set -uo pipefail
MANIFEST=${1:?usage: sbatch --array=0-N hpc/closure_fit_stride.slurm MANIFEST.csv OUTPUT_DIR}
OUTPUT=${2:?usage: sbatch --array=0-N hpc/closure_fit_stride.slurm MANIFEST.csv OUTPUT_DIR}
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p "$OUTPUT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

STRIDE=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
START=${SLURM_ARRAY_TASK_ID:-0}
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
printf 'task %s: manifest %s has %s rows, stride %s\n' "$START" "$MANIFEST" "$ROWS" "$STRIDE"

FAILED=0
for (( i = START; i < ROWS; i += STRIDE )); do
    printf '=== row %s\n' "$i"
    if ! PYTHONPATH="$PWD/contracts/python:$PWD/Coll_Models_v2/src" \
         hpc/python.sh hpc/estimate_closure_task.py \
         --manifest "$MANIFEST" --index "$i" --output "$OUTPUT" \
         --bootstrap "${CLOSURE_BOOTSTRAP:-200}" \
         --propensity-offsets "${CLOSURE_OFFSETS:-64}"; then
        printf 'row %s FAILED\n' "$i" >&2
        FAILED=$(( FAILED + 1 ))
    fi
done
printf 'task %s finished with %s failed rows\n' "$START" "$FAILED"
exit $(( FAILED > 0 ))
