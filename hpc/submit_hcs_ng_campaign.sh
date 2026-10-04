#!/bin/bash
# Submit a staged HCS non-Gaussian campaign with frozen-artifact gating.
#
# Usage:
#   submit_hcs_ng_campaign.sh numerics-pilot ARTIFACT TAG
#   submit_hcs_ng_campaign.sh engineering ARTIFACT TAG NUMERICS_SUMMARY
#   submit_hcs_ng_campaign.sh stability-sentinel ARTIFACT TAG ENGINEERING_SUMMARY
#   submit_hcs_ng_campaign.sh stability ARTIFACT TAG SENTINEL_SUMMARY
#   submit_hcs_ng_campaign.sh sweep ARTIFACT TAG STABILITY_SUMMARY
#   submit_hcs_ng_campaign.sh tails ARTIFACT TAG SWEEP_SUMMARY
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODE=${1:-engineering}
# The released model folder (tag closure-v2-final-v1): the artifact runs with
# the encounter clock, loss and angular tables beside it.
MODEL_VARIANT=${HCS_NG_MODEL_VARIANT:-encounter_final}
ARTIFACT=${2:-models/microscopic_closure_v2_final_v1/closure_v2.npz}
TAG=${3:-${MODE}_${MODEL_VARIANT}_v8}
# Environment overrides outlive a campaign; show them so a stale export from an
# earlier model (e.g. HCS_NG_MODEL_VARIANT=angular_evidence) is never silent.
OVERRIDES=$(env | grep '^HCS_NG_' || true)
[[ -z "$OVERRIDES" ]] || echo "environment overrides: $(echo "$OVERRIDES" | tr '\n' ' ')" >&2
if [[ -f "$(dirname "$ARTIFACT")/model_card.json" && "$MODEL_VARIANT" != "encounter_final" ]]; then
  echo "$ARTIFACT is an encounter-unit model folder but HCS_NG_MODEL_VARIANT=$MODEL_VARIANT;" \
       "run: unset HCS_NG_MODEL_VARIANT" >&2
  exit 2
fi
# Fourth argument: numerics summary for MODE=engineering; engineering summary
# for MODE=stability-sentinel; sentinel summary for MODE=stability; full
# stability summary for sweep/map; passing sweep summary for tails; or
# full-domain HCS summary for domain-pilot.
GATE_SUMMARY=${4:-}
MANIFEST="manifests/hcs_ng_${TAG}.csv"
RESULTS="${HCS_NG_RESULTS_ROOT:-results/validation/hcs}/hcs_ng_${TAG}"
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
if [[ "$MODE" == "numerics-pilot" ]]; then
  CHECK+=(--allow-engineering)
elif [[ "$MODE" == "engineering" || "$MODE" == "stability-sentinel" \
     || "$MODE" == "stability" \
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
# Below the site array limit every array index runs exactly one realization.
# Above it each index strides through several in series, so the per-task
# walltime must cover their sum; say so rather than letting the last stride
# be cut off mid-trajectory.
if (( ROWS > MAX_ARRAY )); then
  echo "note: $ROWS rows exceed MaxArraySize=$MAX_ARRAY; each array task runs" \
       "up to $(( (ROWS + MAX_ARRAY - 1) / MAX_ARRAY )) realizations in series" >&2
fi
CONCURRENT=${HCS_NG_MAX_CORES:-256}; (( CONCURRENT > TASKS )) && CONCURRENT=$TASKS
# The encounter-unit model peaks at 0.98 GiB per process (0.68 GiB for the
# contact-unit model); 1800M keeps ~1.8x headroom and still fits 128 one-core
# tasks on a 256-GiB node, so memory never claims an extra core.
MEMORY=${HCS_NG_MEM_PER_TASK:-1800M}
# Measured protocol-v8 cost: wall time per task is set by the collision count
# and the aspect ratio, not by the particle count.  At fixed box volume the
# number density scales with N, so the physical time to reach a given cpp
# falls as 1/N while the per-step cost rises as N; N=10000 therefore costs
# only 1.1-1.5x N=2000 for the same cpp.  At tau_end=1500 the most expensive
# coordinate (alpha=0.5, AR=1.2) is about 8.4 h, and the half-step sentinel
# arm doubles that.  These defaults carry roughly a 2x margin.  The
# encounter-unit model costs 1.14x per collision (its clock advances 1.19x
# more physical time per collision); measured v8 maxima of 6.2 h (sweep) and
# 7.2 h (sentinel half step) become about 7.0 h and 8.1 h.
case "$MODE" in
  numerics-pilot|engineering) DEFAULT_WALLTIME=06:00:00 ;;
  stability-sentinel) DEFAULT_WALLTIME=24:00:00 ;;
  stability) DEFAULT_WALLTIME=16:00:00 ;;
  domain-pilot) DEFAULT_WALLTIME=04:00:00 ;;
  sweep|map) DEFAULT_WALLTIME=16:00:00 ;;
  tails|sphere-controls) DEFAULT_WALLTIME=20:00:00 ;;
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
