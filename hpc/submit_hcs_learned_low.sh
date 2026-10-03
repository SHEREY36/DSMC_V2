#!/bin/bash
# Validate the far-from-equipartition learned HCS region without USF evidence.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
ARTIFACT=${1:-models/microscopic_closure_v2_usf_candidate/closure_v2.npz}
TAG=${2:-learned_low_v1}
MANIFEST="manifests/hcs_${TAG}.csv"
RESULTS="results/hcs_${TAG}"
PARTICLES=${HCS_PARTICLES:-2000}
TAU_END=${HCS_TAU_END:-120}
REPLICATES=${HCS_REPLICATES:-3}

if [[ -e "$RESULTS/summary.json" ]] || compgen -G "$RESULTS/*.json" >/dev/null; then
  echo "refusing to mix a new campaign with existing outputs in $RESULTS" >&2
  exit 2
fi
mkdir -p logs "$RESULTS"

CASE_ARGS=()
for alpha in 0.5 0.8 0.95 1.0; do
  for ar in 1.1 1.2 1.35; do
    CASE_ARGS+=(--case "$alpha,$ar")
  done
done
hpc/python.sh DSMC_0D_v2/scripts/make_hcs_validation_manifest.py \
  --mode learned-extremes --artifact "$ARTIFACT" \
  --particles "$PARTICLES" --tau-end "$TAU_END" --replicates "$REPLICATES" \
  --theta0 0.0125 --theta0 0.2 "${CASE_ARGS[@]}" \
  --output "$MANIFEST" --results "$RESULTS"

ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
EXPECTED=$(( 12 * 2 * REPLICATES ))
if (( ROWS != EXPECTED )); then
  echo "unexpected learned-low design: $ROWS rows, expected $EXPECTED" >&2
  exit 2
fi
CONCURRENT=$ROWS
(( CONCURRENT > ${HCS_MAX_CORES:-256} )) && CONCURRENT=${HCS_MAX_CORES:-256}
RAW=$(sbatch --parsable --array="0-$((ROWS - 1))%$CONCURRENT" \
  hpc/hcs_validation_array.slurm "$MANIFEST" "$ARTIFACT" false true true)
JOB=${RAW%%;*}
QA_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$JOB" \
  hpc/plot_hcs_validation.slurm "$MANIFEST" \
  "$RESULTS/hcs_theta_attraction.png" "$RESULTS/summary.json")
QA=${QA_RAW%%;*}

echo "hcs_learned_low_job=$JOB ($ROWS one-core tasks; concurrency=$CONCURRENT)"
echo "hcs_learned_low_analysis_job=$QA (afterok:$JOB)"
echo "artifact=$ARTIFACT; corrections=false; exact_initial_temperatures=true"
echo "hcs_rescale_temperature=true; particles=$PARTICLES; tau_end=$TAU_END"
