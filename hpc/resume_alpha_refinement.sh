#!/bin/bash
# Resume an alpha-refinement campaign after the deep QA rejected some nodes.
#
#   TAG=alpha_v1_20261001 bash hpc/resume_alpha_refinement.sh
#
# Refits only the fit-manifest rows whose estimates the last deep QA rejected
# (or that have no estimate), then reruns
#   deep QA -> precompute -> pack -> production gate -> USF + HCS validation -> analysis.
# Finished CTC shards, model tables and gate replays are reused.  The
# near-elastic gate replays (alpha = 0.95 states at alpha 0.9-0.99) are added if
# they are missing.  Set REFIT_ROWS (comma list of fit-manifest indices) to
# override the automatic choice, or REFIT_ROWS=none to skip refitting.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${TAG:?set TAG to the campaign tag}
BOOTSTRAP=${BOOTSTRAP:-200}
MODEL=models/microscopic_closure_v2_${TAG}
ESTIMATES=results/closure_estimates/artifact_grid_alpha_${TAG}
WORK=results/closure_estimates/artifact_precompute_alpha_${TAG}
QA_REPORT=results/closure_estimates/artifact_grid_alpha_${TAG}_validation.json
GRID=manifests/artifact_grid_alpha_${TAG}.csv
FIT=manifests/artifact_grid_alpha_${TAG}_fit.csv
NEAR_MANIFEST=manifests/production_gate_${TAG}_near_elastic.csv
USF_MANIFEST=manifests/usf_encounter_${TAG}.csv
HCS_MANIFEST=manifests/hcs_encounter_${TAG}.csv
for f in "$GRID" "$FIT" "$USF_MANIFEST" "$HCS_MANIFEST" \
         "$MODEL/encounter_cross_section.json" "$MODEL/angular_memory.json"; do
  [[ -f "$f" ]] || { echo "missing $f: was $TAG submitted with hpc/submit_alpha_refinement.sh?" >&2; exit 2; }
done
mkdir -p logs
PY="hpc/python.sh"
export PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src"
id() { printf '%s' "${1%%;*}"; }

# 1. rows to refit: rejected by the last deep QA, or without an estimate
if [[ -z "${REFIT_ROWS:-}" ]]; then
  REFIT_ROWS=$($PY - "$FIT" "$ESTIMATES" "$QA_REPORT" <<'PY'
import csv, json, sys
from pathlib import Path
fit, estimates, report = sys.argv[1:]
rows = list(csv.DictReader(open(fit)))
key = lambda a, t, r: (round(float(a), 6), round(float(t), 6), round(float(r), 6))
bad = set()
if Path(report).is_file():
    qa = json.load(open(report))
    for item in qa.get("qa_failures", []) + qa.get("estimate_failures", []):
        bad.add(key(*item["node"][:3]))
picked = []
for i, row in enumerate(rows):
    target = Path(estimates) / (f"alpha_{float(row['alpha']):.3f}_theta_{float(row['theta']):.3f}_"
                                f"AR_{float(row['aspect_ratio']):.3f}_ensemble_{int(row['ensemble_id']):03d}.json")
    if key(row["alpha"], row["theta"], row["aspect_ratio"]) in bad or not target.is_file():
        picked.append(i)
fitted = {key(r["alpha"], r["theta"], r["aspect_ratio"]) for r in rows}
outside = sorted(bad - fitted)
if outside:
    sys.exit(f"QA rejected nodes that are not in {fit} (reused fits): {outside}")
print(",".join(map(str, picked)) or "none")
PY
)
fi
DEP=()
if [[ "$REFIT_ROWS" != "none" ]]; then
  FITS=$(id "$(sbatch --parsable --array="${REFIT_ROWS}%256" \
    --export=ALL,CLOSURE_FALLBACK_FORM=conditional_logit_cubic_v3,CLOSURE_BOOTSTRAP=$BOOTSTRAP \
    hpc/closure_fit_array.slurm "$FIT" "$ESTIMATES")")
  DEP=(--dependency=afterok:$FITS)
else
  FITS="(none)"
fi

# 2. deep QA of all estimates -> precompute -> pack
QA=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes "${DEP[@]}" \
  hpc/validate_artifact_grid.slurm "$GRID" "$ESTIMATES" "$QA_REPORT")")
GRID_ROWS=$(( $(wc -l < "$GRID") - 1 ))
PRE_TASKS=$GRID_ROWS
MAX_ARRAY=$(scontrol show config 2>/dev/null | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
(( PRE_TASKS > ${MAX_ARRAY:-1000} )) && PRE_TASKS=${MAX_ARRAY:-1000}
PRE=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$QA \
  --array="0-$((PRE_TASKS - 1))%128" \
  hpc/artifact_precompute_stride.slurm "$GRID" "$ESTIMATES" "$WORK" "$GRID_ROWS")")
PACK=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PRE \
  --export=ALL,ARTIFACT_EVENT_UNIT=encounter \
  hpc/aggregate.slurm "$GRID" "$ESTIMATES" "$MODEL" "$WORK")")

# 3. near-elastic gate replays (finished rows are skipped), then the gate
[[ -f "$NEAR_MANIFEST" ]] || $PY hpc/make_gate_replay_manifest.py --tag "$TAG" --source-alphas "" \
  --seed-base 290940000 --output "$NEAR_MANIFEST"
NEAR=$(id "$(sbatch --parsable --array=0-3 hpc/gate_replay_array.slurm "$NEAR_MANIFEST")")
# the gate reads only finished replays, so a failed replay row must not cancel it
GATE=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PACK,afterany:$NEAR \
  hpc/production_gate.slurm "$TAG" "$MODEL")")

# 4. validation on the packed model, then the analysis
TABLE_ENV="ALL,ENCOUNTER_TABLE=$MODEL/encounter_cross_section.json,ANGULAR_MEMORY=$MODEL/angular_memory.json"
USF_ROWS=$(( $(wc -l < "$USF_MANIFEST") - 1 ))
HCS_ROWS=$(( $(wc -l < "$HCS_MANIFEST") - 1 ))
USF_WORKERS=208; (( USF_WORKERS > USF_ROWS )) && USF_WORKERS=$USF_ROWS
USF=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PACK \
  --array="0-$((USF_WORKERS - 1))" --export="$TABLE_ENV,USF_ENC_STRIDE=$USF_WORKERS" \
  hpc/usf_encounter_array.slurm "$USF_MANIFEST" "$MODEL/closure_v2.npz")")
HCS=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PACK \
  --array="0-$((HCS_ROWS - 1))%48" --export="$TABLE_ENV" \
  hpc/hcs_encounter_array.slurm "$HCS_MANIFEST" "$MODEL/closure_v2.npz")")
ANALYSIS=$(id "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterany:$USF:$HCS \
  --export="$TABLE_ENV" hpc/analyze_usf_encounter.slurm "$TAG")")
cat <<MSG
resume of $TAG   model folder $MODEL
  refits          $FITS   (fit-manifest rows: $REFIT_ROWS)
  deep QA         $QA     -> $QA_REPORT
  precompute      $PRE    ($GRID_ROWS nodes)
  pack            $PACK
  near-elastic    $NEAR   (gate replays of the alpha=0.95 states)
  production gate $GATE   -> results/production_gate_${TAG}/gate.{csv,json}
  USF validation  $USF    ($USF_ROWS runs)  HCS validation $HCS ($HCS_ROWS runs)
  analysis        $ANALYSIS -> results/usf_encounter_${TAG}/usf_vs_dem.*  results/hcs_encounter_${TAG}/hcs_vs_dem.csv
MSG
