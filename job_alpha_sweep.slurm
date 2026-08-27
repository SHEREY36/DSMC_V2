#!/bin/bash -l
# Conventional Negishi submission: sbatch job_alpha_sweep.slurm
#
# The array covers 864 routing nodes plus six alpha=1 VSS references.  At most
# 50 nodes run concurrently.  Pilot is the default; override DSMC_STAGE at
# submission time for the separate production shard.
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-869%50
#SBATCH --job-name=ctc_v2
#SBATCH --output=ctc_v2_%A_%a.out
#SBATCH --error=ctc_v2_%A_%a.err

set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$PWD}
if [[ ! -d "$ROOT/HS_CTC_v2" ]]; then
    printf 'ERROR: submit this file from the DSMC_V2 repository root.\n' >&2
    exit 2
fi
cd "$ROOT"

DSMC_STAGE=${DSMC_STAGE:-pilot}
case "$DSMC_STAGE" in
    pilot)
        DEFAULT_SAMPLES=20000
        DEFAULT_SHARD=0
        ;;
    production)
        DEFAULT_SAMPLES=80000
        DEFAULT_SHARD=1
        ;;
    *)
        printf 'ERROR: DSMC_STAGE must be pilot or production, got %s.\n' "$DSMC_STAGE" >&2
        exit 2
        ;;
esac
DSMC_SAMPLES=${DSMC_SAMPLES:-$DEFAULT_SAMPLES}
DSMC_SHARD=${DSMC_SHARD:-$DEFAULT_SHARD}

TASK_INDEX=${SLURM_ARRAY_TASK_ID:?this file must be submitted as a Slurm array}
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
TASK_TMP=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
TASK_MANIFEST="$TASK_TMP/dsmc_v2_${ARRAY_JOB_ID}_${TASK_INDEX}.csv"

hpc/python.sh hpc/make_manifest.py \
    --stage "$DSMC_STAGE" \
    --samples "$DSMC_SAMPLES" \
    --shard "$DSMC_SHARD" \
    --output "$TASK_MANIFEST"

ROW=$(sed -n "$((TASK_INDEX + 2))p" "$TASK_MANIFEST")
if [[ -z "$ROW" ]]; then
    printf 'ERROR: array index %s has no manifest row.\n' "$TASK_INDEX" >&2
    exit 2
fi
IFS=, read -r TASK_ID STAGE ROLE ALPHA THETA AR SEED NSAMPLES SHARD OUTPUT_DIR <<< "$ROW"
if [[ "$TASK_ID" != "$TASK_INDEX" ]]; then
    printf 'ERROR: expected task %s but manifest returned %s.\n' "$TASK_INDEX" "$TASK_ID" >&2
    exit 2
fi

printf 'task=%s stage=%s role=%s alpha=%s theta=%s AR=%s seed=%s samples=%s shard=%s\n' \
    "$TASK_ID" "$STAGE" "$ROLE" "$ALPHA" "$THETA" "$AR" "$SEED" "$NSAMPLES" "$SHARD"

if [[ "${DSMC_DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

if [[ ! -x HS_CTC_v2/build/SphCyl ]]; then
    printf 'ERROR: HS_CTC_v2/build/SphCyl is missing. Run the documented build first.\n' >&2
    exit 2
fi

if [[ -f "$OUTPUT_DIR/_SUCCESS" ]]; then
    printf 'Output already complete; skipping %s.\n' "$OUTPUT_DIR"
    exit 0
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-cores}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd HS_CTC_v2
srun ./build/SphCyl \
    "$ALPHA" "$THETA" 1.0 "$AR" "$ROOT/$OUTPUT_DIR" \
    "$SEED" "$NSAMPLES" v2

PYTHONPATH="$ROOT/contracts/python" \
"$ROOT/hpc/python.sh" scripts/finalize_run.py "$ROOT/$OUTPUT_DIR"
