#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=usf_study_qa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/usf_study_qa_%j.out
#SBATCH --error=logs/usf_study_qa_%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dsmc-v2-usf-matplotlib-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
PYTHONPATH="$ROOT/contracts/python:$ROOT/DSMC_0D_v2/src" \
  hpc/python.sh DSMC_0D_v2/scripts/analyze_usf_study.py \
    --manifest "${1:?manifest required}" \
    --sphere-reference DSMC_0D_v2/reference/usf_sphere_boltzmann_v1.csv \
    --dem-reference DSMC_0D_v2/reference/usf_dem_fresh_v1.csv \
    --output "${2:?summary required}" --figure "${3:?figure required}"
