#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=artifact_pack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/artifact_%j.out
#SBATCH --error=logs/artifact_%j.err

set -euo pipefail
MANIFEST=${1:-manifests/artifact_grid.csv}
ESTIMATES=${2:-results/closure_estimates/artifact_grid}
OUTPUT=${3:-models/microscopic_closure_v2}
PRECOMPUTED=${4:-results/closure_estimates/artifact_precompute}
EXCITATIONS=${5:-}
# Slurm executes a copied script from /var/spool/slurm, so $0 does not point
# into the checkout.  sbatch records the submission directory explicitly.
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
ESTIMATE_ARGS=(--node-estimates "$ROOT/$ESTIMATES")
if [[ -n "$EXCITATIONS" ]]; then
  ESTIMATE_ARGS+=("$ROOT/$EXCITATIONS")
fi
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
"$ROOT/hpc/python.sh" "$ROOT/Coll_Models_v2/scripts/build_artifact.py" \
  --manifest "$ROOT/$MANIFEST" "${ESTIMATE_ARGS[@]}" \
  --output "$ROOT/$OUTPUT" --precomputed-directory "$ROOT/$PRECOMPUTED" \
  --bootstrap 200 \
  --gamma-max-table "$ROOT/Coll_Models_v2/models/legacy_bl/gamma_max_table.json" \
  --one-hit-table "$ROOT/Coll_Models_v2/models/legacy_bl/one_hit_table.json"
