#!/bin/bash -l
# Resume after the already-accepted one-fit-per-element arrays drain.  The
# resumed submitter retries only absent outputs and uses bounded worker arrays.
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=full_model_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/full_model_resume_%j.out
#SBATCH --error=logs/full_model_resume_%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
RESUME_FROM_MISSING=true \
TAG="${TAG:-full_candidate_v1}" \
EXCITATION_MAX_CORES="${EXCITATION_MAX_CORES:-256}" \
ARTIFACT_MAX_CORES="${ARTIFACT_MAX_CORES:-256}" \
bash hpc/submit_full_model_pipeline.sh
