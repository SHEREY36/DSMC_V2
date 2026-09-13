#!/bin/bash
# Resume the USF candidate pipeline from already completed extension fits.
# This never regenerates the 1,296 expensive excitation results.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

GRID=${GRID:-manifests/artifact_grid.csv}
BASELINES=${BASELINES:-results/closure_estimates/artifact_grid}
OLD_EXCITATIONS=${OLD_EXCITATIONS:-results/closure_estimates/excitation_correction-grid}
OLD_SUMMARY=${OLD_SUMMARY:-results/closure_estimates/excitation_correction-grid_summary.json}
OLD_OFFLINE=${OLD_OFFLINE:-results/closure_estimates/excitation_correction-grid_offline_validation.json}
EXT_MANIFEST=${EXT_MANIFEST:-manifests/excitation_usf_extension_v1.csv}
EXT_RESULTS=${EXT_RESULTS:-results/closure_estimates/excitation_usf_extension_v1}
EXT_SUMMARY=${EXT_SUMMARY:-results/closure_estimates/excitation_usf_extension_v1_summary.json}
EXT_OFFLINE=${EXT_OFFLINE:-results/closure_estimates/excitation_usf_extension_v1_offline_validation.json}
ARTIFACT_DIR=${ARTIFACT_DIR:-models/microscopic_closure_v2_usf_candidate}
ARTIFACT=$ARTIFACT_DIR/closure_v2.npz
PRECOMPUTED=${PRECOMPUTED:-results/closure_estimates/artifact_precompute_usf_candidate_v1}
HCS_MANIFEST=${HCS_MANIFEST:-manifests/hcs_corrected_validation_usf_candidate_v1.csv}
HCS_RESULTS=${HCS_RESULTS:-results/hcs_corrected_validation_usf_candidate_v1}
HCS_SUMMARY=$HCS_RESULTS/summary.json
HOLDOUT_MANIFEST=${HOLDOUT_MANIFEST:-manifests/excitation_independent_holdout_v1.csv}
HOLDOUT_BASELINE=${HOLDOUT_BASELINE:-results/closure_estimates/independent_ctc_holdout_v1_baseline/alpha_0.950_theta_1.000_AR_2.000_ensemble_000.json}
HOLDOUT_VALIDATION=${HOLDOUT_VALIDATION:-results/closure_estimates/independent_ctc_holdout_usf_candidate_v1_validation.json}
USF_MANIFEST=${USF_MANIFEST:-manifests/usf_validation_pilot_usf_candidate_v1.csv}
USF_RESULTS=${USF_RESULTS:-results/usf_validation_pilot_usf_candidate_v1}

mkdir -p logs results/closure_estimates
[[ -f "$EXT_MANIFEST" ]] || { echo "missing completed extension manifest: $EXT_MANIFEST" >&2; exit 2; }
[[ -d "$EXT_RESULTS" ]] || { echo "missing completed extension results: $EXT_RESULTS" >&2; exit 2; }
EXT_ROWS=$(( $(wc -l < "$EXT_MANIFEST") - 1 ))
(( EXT_ROWS == 1296 )) || { echo "expected 1296 extension rows, found $EXT_ROWS" >&2; exit 2; }
hpc/python.sh -c '
import csv,json,sys
rows=list(csv.DictReader(open(sys.argv[1], newline="")))
bad=[]
for row in rows:
    try:
        payload=json.load(open(row["output_file"]))
    except (OSError, ValueError) as exc:
        bad.append((row["task_id"], str(exc)))
        continue
    if payload.get("excitation_status") != "pass":
        bad.append((row["task_id"], payload.get("error", "not pass")))
if bad:
    raise SystemExit(f"{len(bad)} extension outputs are absent/invalid; first={bad[0]}")
print(f"reusing {len(rows)} completed extension fits; no excitation rerun")
' "$EXT_MANIFEST"

# Recheck all independent prerequisite evidence before allocating cores.
hpc/python.sh hpc/require_excitation_pass.py "$OLD_SUMMARY" \
  --expected-mode correction-grid
hpc/python.sh -c \
  'import json,sys; assert json.load(open(sys.argv[1]))["offline_validation_pass"]' \
  "$OLD_OFFLINE"
hpc/python.sh -c \
  'import json,sys; assert json.load(open(sys.argv[1]))["validation_pass"]' \
  results/closure_estimates/independent_ctc_holdout_v1_validation.json
PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src" \
  hpc/python.sh hpc/validate_artifact_inputs.py \
    --manifest "$GRID" --estimates "$BASELINES" \
    --output results/closure_estimates/usf_candidate_baseline_validation.json \
    --require-current-estimates --require-pass

# Killed dependencies leave empty directories. They are safe to reuse, but a
# nonempty target could mix different artifacts and is rejected fail-closed.
for TARGET in "$ARTIFACT_DIR" "$PRECOMPUTED" "$HCS_RESULTS" "$USF_RESULTS"; do
  if [[ -e "$TARGET" ]] && find "$TARGET" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "refusing to mix resumed work with existing outputs in $TARGET" >&2
    echo "move that directory or override the corresponding environment variable" >&2
    exit 2
  fi
done
[[ ! -e "$HOLDOUT_VALIDATION" ]] || {
  echo "refusing to overwrite $HOLDOUT_VALIDATION" >&2; exit 2;
}
mkdir -p "$ARTIFACT_DIR" "$PRECOMPUTED" "$HCS_RESULTS" "$USF_RESULTS"

# Recompute QA from the existing files with the repaired material-response and
# fitted-trust-region rules. This is the only stage repeated from the failed run.
QA_RAW=$(sbatch --parsable hpc/validate_usf_extension.slurm \
  "$EXT_MANIFEST" "$EXT_SUMMARY" "$EXT_OFFLINE")
QA_JOB=${QA_RAW%%;*}

MAX_ARRAY=$(scontrol show config 2>/dev/null \
  | awk '$1 == "MaxArraySize" {print $3}') || MAX_ARRAY=1000
MAX_ARRAY=${MAX_ARRAY:-1000}
BASELINE_ROWS=$(( $(wc -l < "$GRID") - 1 ))
PRE_TASKS=$BASELINE_ROWS
(( PRE_TASKS > MAX_ARRAY )) && PRE_TASKS=$MAX_ARRAY
PRE_CONCURRENT=$(( ${ARTIFACT_MAX_CORES:-256} / 2 ))
(( PRE_CONCURRENT > PRE_TASKS )) && PRE_CONCURRENT=$PRE_TASKS
PRE_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$QA_JOB" \
  --array="0-$((PRE_TASKS - 1))%$PRE_CONCURRENT" \
  hpc/artifact_precompute_stride.slurm "$GRID" "$BASELINES" \
  "$PRECOMPUTED" "$BASELINE_ROWS" "$OLD_EXCITATIONS" "$EXT_RESULTS")
PRE_JOB=${PRE_RAW%%;*}
PACK_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PRE_JOB" hpc/aggregate.slurm \
  "$GRID" "$BASELINES" "$ARTIFACT_DIR" "$PRECOMPUTED" \
  "$OLD_EXCITATIONS" "$EXT_RESULTS")
PACK_JOB=${PACK_RAW%%;*}

# HCS and the independent fresh-CTC rescore validate the exact rebuilt bytes.
hpc/python.sh DSMC_0D_v2/scripts/make_hcs_validation_manifest.py \
  --output "$HCS_MANIFEST" --results "$HCS_RESULTS"
HCS_ROWS=$(( $(wc -l < "$HCS_MANIFEST") - 1 ))
HCS_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PACK_JOB" \
  --array="0-$((HCS_ROWS - 1))%$HCS_ROWS" hpc/hcs_validation_array.slurm \
  "$HCS_MANIFEST" "$ARTIFACT" true)
HCS_JOB=${HCS_RAW%%;*}
HCS_PLOT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HCS_JOB" hpc/plot_hcs_validation.slurm \
  "$HCS_MANIFEST" "$HCS_RESULTS/hcs_theta_attraction.png" "$HCS_SUMMARY")
HCS_PLOT_JOB=${HCS_PLOT_RAW%%;*}

HOLDOUT_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$PACK_JOB" hpc/validate_independent_ctc_holdout.slurm \
  "$HOLDOUT_MANIFEST" "$HOLDOUT_BASELINE" "$ARTIFACT" "$HOLDOUT_VALIDATION")
HOLDOUT_JOB=${HOLDOUT_RAW%%;*}
GATE_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$HCS_PLOT_JOB:$HOLDOUT_JOB" \
  hpc/check_usf_candidate.slurm "$HCS_SUMMARY" "$HOLDOUT_VALIDATION" "$ARTIFACT")
GATE_JOB=${GATE_RAW%%;*}

hpc/python.sh DSMC_0D_v2/scripts/make_usf_validation_manifest.py \
  --mode pilot --output "$USF_MANIFEST" --results "$USF_RESULTS"
USF_ROWS=$(( $(wc -l < "$USF_MANIFEST") - 1 ))
USF_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$GATE_JOB" \
  --array="0-$((USF_ROWS - 1))%$USF_ROWS" hpc/usf_validation_array.slurm \
  "$USF_MANIFEST" "$ARTIFACT")
USF_JOB=${USF_RAW%%;*}
USF_QA_RAW=$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:$USF_JOB" hpc/analyze_usf_validation.slurm \
  "$USF_MANIFEST" "$USF_RESULTS/summary.json" "$USF_RESULTS/usf_validation.png")
USF_QA_JOB=${USF_QA_RAW%%;*}

echo "extension_qa_job=$QA_JOB (reuses all $EXT_ROWS completed fits)"
echo "artifact_precompute_job=$PRE_JOB ($PRE_CONCURRENT simultaneous x 2 cores)"
echo "artifact_pack_job=$PACK_JOB (afterok:$PRE_JOB)"
echo "hcs_job=$HCS_JOB; hcs_summary_job=$HCS_PLOT_JOB"
echo "fresh_ctc_rescore_job=$HOLDOUT_JOB (no new CTC trajectories)"
echo "usf_preflight_job=$GATE_JOB"
echo "paired_usf_pilot_job=$USF_JOB; usf_analysis_job=$USF_QA_JOB"
echo "Final pilot decision: $USF_RESULTS/summary.json"
