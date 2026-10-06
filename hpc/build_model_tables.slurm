#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=model_tables
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/model_tables_%j.out
#SBATCH --error=logs/model_tables_%j.err

# Per-node model tables written next to the artifact (one model folder):
#   encounter_cross_section.json  sigma(theta, AR) and <k>(alpha, theta, AR)
#   angular_memory.json           energy-tilted angular law at every node
#   loss_memory.json              loss rates E[eps|z] = c_t z + c_r (1 - z) at every node
set -euo pipefail
MODEL=${1:?model bundle directory required}
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p "$MODEL"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/contracts/python:$PWD/Coll_Models_v2/src:$PWD/DSMC_0D_v2/src"
# the training shards only: holdout, test and v1 shards are never fitted
GLOBS=('results/ctc/nodes/training/*/*')
hpc/python.sh Coll_Models_v2/scripts/build_encounter_table.py \
  --output "$MODEL/encounter_cross_section.json" "${GLOBS[@]}"
hpc/python.sh Coll_Models_v2/scripts/fit_angular_memory_table.py \
  --workers "${SLURM_CPUS_PER_TASK:-64}" --output "$MODEL/angular_memory.json" "${GLOBS[@]}"
hpc/python.sh Coll_Models_v2/scripts/fit_loss_memory_table.py \
  --workers "${SLURM_CPUS_PER_TASK:-64}" --output "$MODEL/loss_memory.json" "${GLOBS[@]}"
