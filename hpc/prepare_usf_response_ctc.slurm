#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=ctc_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/ctc_prepare_%j.out
#SBATCH --error=logs/ctc_prepare_%j.err
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd "$ROOT"
hpc/python.sh hpc/make_usf_response_ctc_manifest.py \
  --harvest-manifest "${1:?harvest manifest}" --output "${2:?CTC manifest}" \
  --results "${3:?CTC results root}" --replay-samples "${CTC_REPLAY_SAMPLES:-40000}"
