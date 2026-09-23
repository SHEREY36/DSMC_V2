#!/bin/bash
# Submit a staged HCS non-Gaussian campaign with frozen-artifact gating.
#
# Usage:
#   submit_hcs_ng_campaign.sh engineering ARTIFACT TAG
#   submit_hcs_ng_campaign.sh stability-sentinel ARTIFACT TAG ENGINEERING_SUMMARY
#   submit_hcs_ng_campaign.sh stability ARTIFACT TAG SENTINEL_SUMMARY
#   submit_hcs_ng_campaign.sh sweep ARTIFACT TAG STABILITY_SUMMARY
#   submit_hcs_ng_campaign.sh tails ARTIFACT TAG SWEEP_SUMMARY
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-engineering}
MODEL_VARIANT=${HCS_NG_MODEL_VARIANT:-angular_evidence}
ARTIFACT=${2:-models/microscopic_closure_v2_angular_evidence/closure_v2.npz}
TAG=${3:-${MODE}_${MODEL_VARIANT}_v4}
# Fourth argument: engineering summary for MODE=stability-sentinel; sentinel
# summary for MODE=stability; full stability summary for sweep/map; passing
# sweep summary for tails; or full-domain HCS summary for domain-pilot.
GATE_SUMMARY=${4:-}
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
  --mode "$MODE" --model-variant "$MODEL_VARIANT" --artifact "$ARTIFACT" \
  --output "$MANIFEST" --results "$RESULTS"
CHECK=(--manifest "$MANIFEST" --artifact "$ARTIFACT")
if [[ "$MODE" == "engineering" ]]; then
  CHECK+=(--allow-engineering)
elif [[ "$MODE" == "stability-sentinel" || "$MODE" == "stability" \
     || "$MODE" == "sweep" \
     || "$MODE" == "map" || "$MODE" == "tails" ]]; then
  [[ -n "$GATE_SUMMARY" ]] || {
    echo "$MODE requires a gate summary as argument 4" >&2
    exit 2
  }
  CHECK+=(--pilot-summary "$GATE_SUMMARY")
elif [[ -n "$GATE_SUMMARY" ]]; then
  CHECK+=(--hcs-summary "$GATE_SUMMARY")
fi
PYTHONPATH="$ROOT/DSMC_0D_v2/src" hpc/python.sh hpc/check_hcs_ng_prerequisites.py "${CHECK[@]}"
ROWS=$(( $(wc -l < "$MANIFEST") - 1 ))
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
MAX_ARRAY=${MAX_ARRAY:-1000}
TASKS=$ROWS; (( TASKS > MAX_ARRAY )) && TASKS=$MAX_ARRAY
CONCURRENT=${HCS_NG_MAX_CORES:-256}; (( CONCURRENT > TASKS )) && CONCURRENT=$TASKS
MEMORY=${HCS_NG_MEM_PER_TASK:-1500M}
case "$MODE" in
  engineering) DEFAULT_WALLTIME=02:00:00 ;;
  stability-sentinel|stability) DEFAULT_WALLTIME=24:00:00 ;;
  domain-pilot) DEFAULT_WALLTIME=04:00:00 ;;
  sweep|map) DEFAULT_WALLTIME=08:00:00 ;;
  tails|sphere-controls) DEFAULT_WALLTIME=10:00:00 ;;
esac
WALLTIME=${HCS_NG_WALLTIME:-$DEFAULT_WALLTIME}
QOS=${HCS_NG_QOS:-normal}
RAW=$(sbatch --parsable --array="0-$((TASKS - 1))%$CONCURRENT" \
  --mem="$MEMORY" --time="$WALLTIME" --qos="$QOS" \
  --export="ALL,HCS_NG_STRIDE=$TASKS" hpc/hcs_ng_array.slurm "$MANIFEST" "$ARTIFACT")
JOB=${RAW%%;*}
# afterany: a partial campaign must still be summarized (missing tasks are
# listed and force a false verdict) instead of the QA job silently vanishing.
QA_RAW=$(sbatch --parsable --dependency="afterany:$JOB" \
  hpc/analyze_hcs_ng.slurm "$MANIFEST" "$SUMMARY" "$FIGURE")
QA=${QA_RAW%%;*}
echo "hcs_ng_job=$JOB ($ROWS virtual tasks; concurrency=$CONCURRENT)"
echo "hcs_ng_analysis_job=$QA (afterany:$JOB)"
echo "model=$MODEL_VARIANT artifact=$ARTIFACT memory/task=$MEMORY walltime=$WALLTIME qos=$QOS"
echo "Inspect $SUMMARY before submitting a larger mode."
