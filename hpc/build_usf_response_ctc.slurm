#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=ctc_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/ctc_build_%j.out
#SBATCH --error=logs/ctc_build_%j.err
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT/HS_CTC_v2/build"
make clean
make -j "${SLURM_CPUS_PER_TASK:-8}"
test -x SphCyl
