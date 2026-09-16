#!/bin/bash
# Submit a staged HCS non-Gaussian campaign with frozen-artifact gating.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-engineering}
ARTIFACT=${2:-models/microscopic_closure_v2_candidate/closure_v2.npz}
TAG=${3:-${MODE}_v1}
HCS_SUMMARY=${4:-}
MANIFEST="manifests/hcs_ng_${TAG}.csv"
RESULTS="results/hcs_ng_${TAG}"
SUMMARY="$RESULTS/summary.json"
FIGURE="$RESULTS/non_gaussian_observables.png"
if [[ -e "$SUMMARY" ]] || compgen -G "$RESULTS/*.txt" >/dev/null; then
  echo "refusing to mix a new campaign with existing outputs in $RESULTS" >&2
  exit 2
fi
mkdir -p logs "$RESULTS"
PYTHONPATH="$ROOT/DSMC_0D_v2/src" hpc/python.sh \
  DSMC_0D_v2/scripts/make_hcs_ng_manifest.py \
  --mode "$MODE" --artifact "$ARTIFACT" --output "$MANIFEST" --results "$RESULTS"
CHECK=(--manifest "$MANIFEST" --artifact "$ARTIFACT")
if [[ "$MODE" == "engineering" ]]; then
  CHECK+=(--allow-engineering)
elif [[ -n "$HCS_SUMMARY" ]]; then
  CHECK+=(--hcs-summary "$HCS_SUMMARY")
fi
PYTHONPATH="$ROOT/DSMC_0D_v2/src" hpc/python.sh hpc/check_hcs_ng_prerequisites.py "${CHECK[@]}"
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
MAX_ARRAY=${MAX_ARRAY:-1000}
TASKS=$ROWS; (( TASKS > MAX_ARRAY )) && TASKS=$MAX_ARRAY
CONCURRENT=${HCS_NG_MAX_CORES:-256}; (( CONCURRENT > TASKS )) && CONCURRENT=$TASKS
RAW=$(sbatch --parsable --array="0-$((TASKS - 1))%$CONCURRENT" \
  --export="ALL,HCS_NG_STRIDE=$TASKS" hpc/hcs_ng_array.slurm "$MANIFEST" "$ARTIFACT")
JOB=${RAW%%;*}
QA_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$JOB" \
  hpc/analyze_hcs_ng.slurm "$MANIFEST" "$SUMMARY" "$FIGURE")
QA=${QA_RAW%%;*}
echo "hcs_ng_job=$JOB ($ROWS virtual tasks; concurrency=$CONCURRENT)"
echo "hcs_ng_analysis_job=$QA (afterok:$JOB)"
echo "Inspect $SUMMARY before submitting a larger mode."
