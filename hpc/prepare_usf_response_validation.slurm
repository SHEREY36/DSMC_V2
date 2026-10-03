#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=resp_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/resp_prepare_%j.out
#SBATCH --error=logs/resp_prepare_%j.err
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}; cd "$ROOT"
ARTIFACT=${1:?candidate artifact}; HCS_MANIFEST=${2:?HCS manifest}
HCS_RESULTS=${3:?HCS results}; USF_MANIFEST=${4:?USF manifest}
USF_RESULTS=${5:?USF results}
[[ -f "$ARTIFACT" ]] || { echo "candidate artifact missing" >&2; exit 2; }
hpc/python.sh DSMC_0D_v2/scripts/make_hcs_validation_manifest.py \
  --mode full-domain-extremes --artifact "$ARTIFACT" --output "$HCS_MANIFEST" \
  --results "$HCS_RESULTS" --particles 5000 --tau-end 20 --replicates 2 \
  --state-update-cpp 0.05
hpc/python.sh DSMC_0D_v2/scripts/make_usf_study_manifest.py \
  --mode response-validation --artifact "$ARTIFACT" --output "$USF_MANIFEST" \
  --results "$USF_RESULTS" --particles 30000 --replicates 2 \
  --volume-fraction 0.01
[[ $(( $(wc -l < "$HCS_MANIFEST") - 1 )) -eq 232 ]] || {
  echo "unexpected HCS validation row count" >&2; exit 2; }
[[ $(( $(wc -l < "$USF_MANIFEST") - 1 )) -eq 276 ]] || {
  echo "unexpected USF validation row count" >&2; exit 2; }
