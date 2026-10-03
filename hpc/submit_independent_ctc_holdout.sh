#!/bin/bash
# Fresh-trajectory sentinel for the correction Jacobian; no training shard is reused.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
ARTIFACT=${1:-models/microscopic_closure_v2_candidate/closure_v2.npz}
HCS_SUMMARY=${2:-results/hcs_corrected_validation_domain_v3/summary.json}
TAG=${3:-v1}
CTC_MANIFEST="manifests/independent_ctc_holdout_${TAG}.csv"
FIT_MANIFEST="manifests/independent_ctc_holdout_fit_${TAG}.csv"
CTC_RESULTS="results/ctc_independent_holdout_${TAG}"
ESTIMATES="results/closure_estimates/independent_ctc_holdout_${TAG}_baseline"
EXCITATION_MANIFEST="manifests/excitation_independent_holdout_${TAG}.csv"
EXCITATION_RESULTS="results/closure_estimates/excitation_independent_holdout_${TAG}"
VALIDATION="results/closure_estimates/independent_ctc_holdout_${TAG}_validation.json"

[[ -f "$ARTIFACT" ]] || { echo "candidate artifact missing: $ARTIFACT" >&2; exit 2; }
mkdir -p logs "$ESTIMATES" "$EXCITATION_RESULTS"
hpc/python.sh hpc/require_hcs_pass.py "$HCS_SUMMARY"
hpc/python.sh hpc/make_independent_ctc_holdout_manifest.py \
  --ctc-output "$CTC_MANIFEST" \
  --fit-output "$FIT_MANIFEST" \
  --results-root "$CTC_RESULTS"

# Two 200k-hit CTC jobs integrate genuinely fresh collision trajectories.  The
# first is the inelastic validation node; the second supplies its independent
# elastic equilibrium anchor.
CTC_RAW=$(sbatch --parsable --array=0-1%2 hpc/ctc_stride.slurm "$CTC_MANIFEST")
CTC_JOB=${CTC_RAW%%;*}

# The geometric propensity is deterministic. Twenty threads split disjoint
# event blocks and write one cache that every excitation fit then reads.
BASE_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$CTC_JOB" --array=0 \
  --cpus-per-task=20 --mem=40G \
  --export=ALL,CLOSURE_BOOTSTRAP=20,CLOSURE_OFFSETS=128,CLOSURE_PROPENSITY_WORKERS=20 \
  hpc/closure_fit_array.slurm "$FIT_MANIFEST" "$ESTIMATES")
BASE_JOB=${BASE_RAW%%;*}

PREP_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$BASE_JOB" hpc/prepare_independent_ctc_holdout.slurm \
  "$FIT_MANIFEST" "$ESTIMATES" "$EXCITATION_MANIFEST" "$EXCITATION_RESULTS")
PREP_JOB=${PREP_RAW%%;*}

# Bootstrap is deliberately zero here: it does not change any point fit, and
# the gate compares 36 independent boundary responses. If a response fails,
# only that direction is rerun later with extra bootstrap uncertainty rather
# than spending hundreds of unnecessary fit-hours in advance.
FIT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PREP_JOB" --array=0-35%36 \
  --export=ALL,EXCITATION_BOOTSTRAP=0,EXCITATION_OFFSETS=128 \
  hpc/excitation_fit_stride.slurm "$EXCITATION_MANIFEST")
FIT_JOB=${FIT_RAW%%;*}

BASELINE="$ESTIMATES/alpha_0.950_theta_1.000_AR_2.000_ensemble_000.json"
QA_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$FIT_JOB" hpc/validate_independent_ctc_holdout.slurm \
  "$EXCITATION_MANIFEST" "$BASELINE" "$ARTIFACT" "$VALIDATION")
QA_JOB=${QA_RAW%%;*}

echo "fresh_ctc_job=$CTC_JOB (2 tasks x 20 cores)"
echo "fresh_baseline_fit_job=$BASE_JOB (afterok:$CTC_JOB; 20 propensity workers)"
echo "holdout_prepare_job=$PREP_JOB (afterok:$BASE_JOB)"
echo "holdout_fit_job=$FIT_JOB (36 point fits; afterok:$PREP_JOB)"
echo "holdout_validation_job=$QA_JOB (afterok:$FIT_JOB)"
echo "Final decision file: $VALIDATION"
