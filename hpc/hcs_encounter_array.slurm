#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=hcs_enc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1800M
#SBATCH --time=06:00:00
#SBATCH --output=logs/hcs_enc_%A_%a.out
#SBATCH --error=logs/hcs_enc_%A_%a.err

set -euo pipefail
MANIFEST=${1:?manifest required}
ARTIFACT=${2:?artifact required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
INDEX=${SLURM_ARRAY_TASK_ID:?}
ARM=$(awk -F, -v row=$((INDEX + 2)) 'NR == row {print $3}' "$MANIFEST")
MODEL=()
case "$ARM" in
  contact_legacy) MODEL=(--event-unit contact_legacy) ;;
  encounter_memory) MODEL=(--event-unit encounter
                           --angular-memory "${ANGULAR_MEMORY:-DSMC_0D_v2/models/angular_memory_v1.json}"
                           ${LOSS_MEMORY:+--loss-memory "$LOSS_MEMORY"}
                           --encounter-table "${ENCOUNTER_TABLE:-DSMC_0D_v2/models/encounter_cross_section_v2.json}") ;;
  *) echo "unknown arm '$ARM' in row $INDEX" >&2; exit 2 ;;
esac
PREFIX=$(awk -F, -v row=$((INDEX + 2)) 'NR == row {print $NF}' "$MANIFEST")
mkdir -p "$(dirname "$PREFIX")"
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src" \
  hpc/python.sh DSMC_0D_v2/scripts/run_hcs_validation_task.py \
    --manifest "$MANIFEST" --task "$INDEX" --artifact "$ARTIFACT" \
    --exact-initial-temperatures --rescale-hcs-temperature "${MODEL[@]}"
