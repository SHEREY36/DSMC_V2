#!/bin/bash -l
# One independent artifact node per work item.  The submitter allows 128
# two-core tasks to run concurrently, exactly filling the account's 256
# purchased Negishi cores.  Extra array elements start as soon as slots free;
# the stride handles grids larger than the cluster's MaxArraySize.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=artifact_pre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/artifact_pre_%A_%a.out
#SBATCH --error=logs/artifact_pre_%A_%a.err

set -euo pipefail
MANIFEST=${1:?manifest path is required}
ESTIMATES=${2:?node-estimate directory is required}
WORK=${3:?precompute output directory is required}
ROWS=${4:?baseline node count is required}
EXCITATIONS=${5:-}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"

# The geometry routine owns exactly SLURM_CPUS_PER_TASK threads.  Keep BLAS
# libraries single-threaded so 128 array tasks cannot oversubscribe the nodes.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
STRIDE=$(( ${SLURM_ARRAY_TASK_MAX:?} - ${SLURM_ARRAY_TASK_MIN:?} + 1 ))
INDEX=${SLURM_ARRAY_TASK_ID:?}

while (( INDEX < ROWS )); do
  ESTIMATE_ARGS=(--node-estimates "$ESTIMATES")
  if [[ -n "$EXCITATIONS" ]]; then
    ESTIMATE_ARGS+=("$EXCITATIONS")
  fi
  PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
    hpc/python.sh hpc/precompute_artifact_task.py \
      --manifest "$MANIFEST" "${ESTIMATE_ARGS[@]}" \
      --output "$WORK" --index "$INDEX" \
      --workers "${SLURM_CPUS_PER_TASK:-2}" --propensity-offsets 128 \
      --gamma-max-table Coll_Models_v2/models/legacy_bl/gamma_max_table.json \
      --one-hit-table Coll_Models_v2/models/legacy_bl/one_hit_table.json
  INDEX=$(( INDEX + STRIDE ))
done
