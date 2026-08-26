#!/bin/bash
#SBATCH --account=morri353
#SBATCH --job-name=estimate_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/estimate_%A_%a.out
#SBATCH --error=logs/estimate_%A_%a.err

set -euo pipefail
MANIFEST=${1:?manifest path is required}
OUTPUT=${2:-coefficients/node_estimates}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
"$ROOT/hpc/python.sh" "$ROOT/Coll_Models_v2/scripts/estimate_manifest_row.py" \
  --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --output "$ROOT/$OUTPUT" --bootstrap 2000 \
  --gamma-max-table "$ROOT/Coll_Models_v2/models/legacy_bl/gamma_max_table.json" \
  --one-hit-table "$ROOT/Coll_Models_v2/models/legacy_bl/one_hit_table.json"
