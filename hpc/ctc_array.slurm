#!/bin/bash -l
#SBATCH --account=morri353
#SBATCH --partition=cpu
#SBATCH --job-name=ctc_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ctc_%A_%a.out
#SBATCH --error=logs/ctc_%A_%a.err

set -euo pipefail
MANIFEST=${1:?manifest path is required}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=, read -r TASK_ID STAGE ROLE ALPHA THETA AR SEED NSAMPLES SHARD OUTPUT_DIR <<< "$ROW"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
cd "$ROOT/HS_CTC_v2"
srun ./build/SphCyl "$ALPHA" "$THETA" 1.0 "$AR" "$ROOT/$OUTPUT_DIR" "$SEED" "$NSAMPLES" v2
PYTHONPATH="$ROOT/contracts/python" \
"$ROOT/hpc/python.sh" scripts/finalize_run.py "$ROOT/$OUTPUT_DIR"
