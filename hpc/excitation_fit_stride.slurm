#!/bin/bash -l
# Independent virtual excitation fits.  Each array element processes a
# balanced stride through the manifest, keeping the number of submitted jobs
# small while retaining 256-way concurrency.  No fresh CTC is generated.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=excitation_fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/excitation_fit_%A_%a.out
#SBATCH --error=logs/excitation_fit_%A_%a.err

set -euo pipefail
MANIFEST=${1:?excitation manifest is required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
STRIDE=$(( ${SLURM_ARRAY_TASK_MAX:?} - ${SLURM_ARRAY_TASK_MIN:?} + 1 ))
if [[ -n "${EXCITATION_START:-}" ]]; then
  INDEX=$(( EXCITATION_START + ${SLURM_ARRAY_TASK_ID:?} ))
  STOP=${EXCITATION_STOP:?EXCITATION_STOP is required with EXCITATION_START}
else
  INDEX=${SLURM_ARRAY_TASK_ID:?}
  STOP=$ROWS
fi

while (( INDEX < STOP && INDEX < ROWS )); do
  STATUS=1
  for ATTEMPT in $(seq 1 "${EXCITATION_IMPORT_RETRIES:-3}"); do
    if PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
      hpc/python.sh hpc/run_excitation_task.py \
        --manifest "$MANIFEST" --index "$INDEX" \
        --bootstrap "${EXCITATION_BOOTSTRAP:-50}" \
        --propensity-offsets "${EXCITATION_OFFSETS:-128}"; then
      STATUS=0
      break
    else
      STATUS=$?
      echo "excitation row $INDEX failed before publishing output; import/runtime retry $ATTEMPT" >&2
      sleep $(( 5 * ATTEMPT ))
    fi
  done
  (( STATUS == 0 )) || exit "$STATUS"
  INDEX=$(( INDEX + STRIDE ))
done
