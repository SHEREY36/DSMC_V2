#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=gate_ctc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gate_ctc_%A_%a.out
#SBATCH --error=logs/gate_ctc_%A_%a.err

# Strided worker over the gate replay manifest; completed rows are skipped, so
# resubmitting the same array is the restart.
set -uo pipefail
MANIFEST=${1:?manifest required}
cd "${SLURM_SUBMIT_DIR:?}"
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
STRIDE=$(( ${SLURM_ARRAY_TASK_MAX:?} - ${SLURM_ARRAY_TASK_MIN:?} + 1 ))
FAILED=0
for (( i = SLURM_ARRAY_TASK_ID; i < ROWS; i += STRIDE )); do
  bash hpc/run_gate_replay_row.sh "$MANIFEST" "$i" || { echo "row $i FAILED" >&2; FAILED=$((FAILED + 1)); }
done
exit $(( FAILED > 0 ))
