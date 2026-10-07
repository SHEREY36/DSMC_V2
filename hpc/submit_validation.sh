#!/bin/bash
# Validate a finished model folder: production gate (diagnostic), USF against
# DEM (320 runs) and HCS at every corrected-DEM coordinate (130 runs).  No fits.
#
#   TAG=C1_20261005_val MODEL=models/microscopic_closure_v2_C1_20261005 \
#     HCS_REFERENCE=DSMC_0D_v2/reference/hcs_dem_C1.csv \
#     USF_BENCHMARK=DSMC_0D_v2/reference/usf_benchmark_C1.csv bash hpc/submit_validation.sh
#
# The DEM references are required (one contact model per comparison; Model R:
# hcs_dem_fresh_v1.csv and usf_dem_and_legacy_v1.csv).  The production gate replays
# results/ctc/replays/gate_$REPLAY_TAG and is skipped when those replays are absent.
#
# QOS=standby uses idle cores instead of the group's 256.  Runs that already
# finished are skipped, so resubmitting with the same TAG is the restart.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${TAG:?set TAG for the validation outputs}
MODEL=${MODEL:?set MODEL to the model folder}
HCS_REFERENCE=${HCS_REFERENCE:?set HCS_REFERENCE to the DEM HCS table}
USF_BENCHMARK=${USF_BENCHMARK:?set USF_BENCHMARK to the DEM USF table}
[[ -f "$HCS_REFERENCE" && -f "$USF_BENCHMARK" ]] || { echo "missing DEM reference table" >&2; exit 2; }
QOS=${QOS:-normal}
REPLAY_TAG=${REPLAY_TAG:-alpha_v1_20261001}
USF_MANIFEST=manifests/usf_encounter_${TAG}.csv
HCS_MANIFEST=manifests/hcs_encounter_${TAG}.csv
for f in closure_v2.npz encounter_cross_section.json angular_memory.json loss_memory.json; do
  [[ -f "$MODEL/$f" ]] || { echo "missing $MODEL/$f" >&2; exit 2; }
done
mkdir -p logs manifests
PY="hpc/python.sh"
export PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src"
$PY hpc/write_model_card.py "$MODEL"
[[ -f "$USF_MANIFEST" ]] || $PY DSMC_0D_v2/scripts/make_usf_encounter_manifest.py --tag "$TAG" \
    --ablation-replicates 0 --output "$USF_MANIFEST"
[[ -f "$HCS_MANIFEST" ]] || $PY DSMC_0D_v2/scripts/make_hcs_encounter_manifest.py --tag "$TAG" \
    --arms encounter_memory --dem-reference "$HCS_REFERENCE" \
    --replicates 2 --tau-end 100 --output "$HCS_MANIFEST"
USF_ROWS=$(( $(wc -l < "$USF_MANIFEST") - 1 ))
HCS_ROWS=$(( $(wc -l < "$HCS_MANIFEST") - 1 ))
LONG=12:00:00; [[ "$QOS" == standby ]] && LONG=04:00:00
id() { printf '%s' "${1%%;*}"; }
TABLE_ENV="ALL,ENCOUNTER_TABLE=$MODEL/encounter_cross_section.json,ANGULAR_MEMORY=$MODEL/angular_memory.json,LOSS_MEMORY=$MODEL/loss_memory.json"
GATE="skipped (no replays results/ctc/replays/gate_${REPLAY_TAG})"
if ls -d results/ctc/replays/gate_${REPLAY_TAG}/* >/dev/null 2>&1; then
  GATE=$(id "$(sbatch --parsable --qos=normal --time=02:00:00 --export=ALL,REPLAY_TAG=$REPLAY_TAG \
    hpc/production_gate.slurm "$TAG" "$MODEL")")
fi
USF=$(id "$(sbatch --parsable --qos="$QOS" --time=$LONG --array="0-$((USF_ROWS - 1))" \
  --export="$TABLE_ENV,USF_ENC_STRIDE=$USF_ROWS" hpc/usf_encounter_array.slurm "$USF_MANIFEST" "$MODEL/closure_v2.npz")")
HCS=$(id "$(sbatch --parsable --qos="$QOS" --time=$LONG --array="0-$((HCS_ROWS - 1))" \
  --export="$TABLE_ENV" hpc/hcs_encounter_array.slurm "$HCS_MANIFEST" "$MODEL/closure_v2.npz")")
ANALYSIS=$(id "$(sbatch --parsable --qos=normal --dependency=afterany:$USF:$HCS \
  --export="$TABLE_ENV,USF_BENCHMARK=$USF_BENCHMARK" hpc/analyze_usf_encounter.slurm "$TAG")")
cat <<MSG
validation $TAG of $MODEL on QOS $QOS
  production gate  $GATE      -> results/validation/production_gate/${TAG}/gate.{csv,json}
  USF              $USF       ($USF_ROWS runs)
  HCS              $HCS       ($HCS_ROWS runs)
  analysis         $ANALYSIS  -> results/validation/usf/usf_encounter_${TAG}/usf_vs_dem.*
                                 results/validation/hcs/hcs_encounter_${TAG}/hcs_vs_dem.csv
MSG
