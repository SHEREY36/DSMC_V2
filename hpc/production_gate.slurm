#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=prod_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prod_gate_%j.out
#SBATCH --error=logs/prod_gate_%j.err

# Production gate: model collisional productions versus exact CTC on the
# replayed USF pairs, at every fitted alpha plane and between planes.
set -euo pipefail
TAG=${1:?tag required}
MODEL=${2:?model bundle directory required}
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH="$PWD/contracts/python:$PWD/Coll_Models_v2/src:$PWD/DSMC_0D_v2/src" \
  hpc/python.sh DSMC_0D_v2/scripts/production_gate.py \
    --artifact "$MODEL/closure_v2.npz" --angular-memory "$MODEL/angular_memory.json" \
    --encounter-table "$MODEL/encounter_cross_section.json" \
    $( [[ -f "$MODEL/loss_memory.json" ]] && echo --loss-memory "$MODEL/loss_memory.json" ) \
    --workers "${SLURM_CPUS_PER_TASK:-32}" --samples 40000 \
    --output "results/validation/production_gate/${TAG}/gate" results/ctc/replays/gate_${TAG}/*
