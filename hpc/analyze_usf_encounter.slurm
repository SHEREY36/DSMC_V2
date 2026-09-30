#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=enc_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/enc_analysis_%j.out
#SBATCH --error=logs/enc_analysis_%j.err

set -euo pipefail
TAG=${1:?tag required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
export PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src"
hpc/python.sh DSMC_0D_v2/scripts/analyze_usf_encounter.py \
  --results "results/usf_encounter_${TAG}" \
  --output "results/usf_encounter_${TAG}/usf_vs_dem.csv"
hpc/python.sh DSMC_0D_v2/scripts/analyze_hcs_encounter.py \
  --manifest "manifests/hcs_encounter_${TAG}.csv" \
  --output "results/hcs_encounter_${TAG}/hcs_vs_dem.csv"
