#!/bin/bash
#SBATCH --account=morri353
#SBATCH --job-name=artifact_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#SBATCH --output=logs/artifact_%j.out
#SBATCH --error=logs/artifact_%j.err

set -euo pipefail
RUNS_ROOT=${1:-results/ctc}
OUTPUT=${2:-models/collision_operator_v2}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
python3 "$ROOT/Coll_Models_v2/scripts/build_artifact.py" \
  --runs-root "$ROOT/$RUNS_ROOT" --output "$ROOT/$OUTPUT" --bootstrap 2000
