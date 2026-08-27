#!/bin/bash -l
# Build the DSMC runtime bundle only after every combined QA row passes.
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=artifact_v2
#SBATCH --output=artifact_v2_%j.out
#SBATCH --error=artifact_v2_%j.err

set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
hpc/python.sh hpc/require_qa_pass.py coefficients/combined/qa_summary.csv --expected 870
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
exec hpc/python.sh Coll_Models_v2/scripts/build_artifact.py \
    --runs-root results/ctc --node-estimates coefficients/combined_nodes \
    --output models/microscopic_closure_v2 --bootstrap 2000 \
    --gamma-max-table Coll_Models_v2/models/legacy_bl/gamma_max_table.json \
    --one-hit-table Coll_Models_v2/models/legacy_bl/one_hit_table.json
