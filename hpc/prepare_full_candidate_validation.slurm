#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=prepare_gates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/prepare_gates_%j.out
#SBATCH --error=logs/prepare_gates_%j.err

set -euo pipefail
ARTIFACT=${1:?artifact required}
TAG=${2:?tag required}
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
hpc/python.sh hpc/require_full_domain_correction.py "$ARTIFACT"
hpc/python.sh DSMC_0D_v2/scripts/make_hcs_validation_manifest.py \
  --mode full-domain --artifact "$ARTIFACT" \
  --output "manifests/hcs_${TAG}_reference.csv" \
  --results "results/hcs_${TAG}_reference" --state-update-cpp 0
hpc/python.sh DSMC_0D_v2/scripts/make_hcs_validation_manifest.py \
  --mode full-domain --artifact "$ARTIFACT" \
  --output "manifests/hcs_${TAG}_optimized.csv" \
  --results "results/hcs_${TAG}_full_domain" --state-update-cpp 0.05
hpc/python.sh DSMC_0D_v2/scripts/make_usf_validation_manifest.py \
  --mode pilot --output "manifests/usf_${TAG}_pilot.csv" \
  --results "results/usf_${TAG}_pilot" --state-update-cpp 0.05
hpc/python.sh DSMC_0D_v2/scripts/make_usf_validation_manifest.py \
  --mode full --output "manifests/usf_${TAG}_full.csv" \
  --results "results/usf_${TAG}_full" --state-update-cpp 0.05
PYTHONPATH="$ROOT/DSMC_0D_v2/src" hpc/python.sh \
  DSMC_0D_v2/scripts/make_hcs_ng_manifest.py --mode domain-pilot \
  --artifact "$ARTIFACT" --output "manifests/hcs_ng_${TAG}_domain_pilot.csv" \
  --results "results/hcs_ng_${TAG}_domain_pilot"
