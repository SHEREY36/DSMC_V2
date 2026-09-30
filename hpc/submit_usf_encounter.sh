#!/bin/bash
# One-command submission of the encounter-unit USF + HCS validation campaign.
#   TAG=usf_encounter_v1_20260930 bash hpc/submit_usf_encounter.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TAG=${TAG:?set TAG to a fresh campaign tag}
ARTIFACT=${ARTIFACT:-models/microscopic_closure_v2_angular_evidence/closure_v2.npz}
MAX_CORES=${MAX_CORES:-256}
HCS_CORES=${HCS_CORES:-48}
USF_MANIFEST="manifests/usf_encounter_${TAG}.csv"
HCS_MANIFEST="manifests/hcs_encounter_${TAG}.csv"
if [[ -e "$USF_MANIFEST" || -d "results/usf_encounter_${TAG}" ]]; then
  echo "tag $TAG already used; pick a fresh TAG (tasks skip completed rows if you resubmit the arrays by hand)" >&2
  exit 2
fi
for f in "$ARTIFACT" DSMC_0D_v2/models/encounter_cross_section_v2.json \
         DSMC_0D_v2/models/angular_memory_v1.json DSMC_0D_v2/reference/usf_dem_and_legacy_v1.csv; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p logs manifests
export PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src"
hpc/python.sh DSMC_0D_v2/scripts/make_usf_encounter_manifest.py --tag "$TAG" --output "$USF_MANIFEST"
hpc/python.sh DSMC_0D_v2/scripts/make_hcs_encounter_manifest.py --tag "$TAG" --output "$HCS_MANIFEST"
USF_ROWS=$(( $(wc -l < "$USF_MANIFEST") - 1 ))
HCS_ROWS=$(( $(wc -l < "$HCS_MANIFEST") - 1 ))
USF_WORKERS=$(( MAX_CORES - HCS_CORES ))
(( USF_WORKERS > USF_ROWS )) && USF_WORKERS=$USF_ROWS
USF_JOB=$(sbatch --parsable --array="0-$((USF_WORKERS - 1))" \
  --export="ALL,USF_ENC_STRIDE=$USF_WORKERS" hpc/usf_encounter_array.slurm "$USF_MANIFEST" "$ARTIFACT")
HCS_JOB=$(sbatch --parsable --array="0-$((HCS_ROWS - 1))%${HCS_CORES}" \
  hpc/hcs_encounter_array.slurm "$HCS_MANIFEST" "$ARTIFACT")
ANALYSIS_JOB=$(sbatch --parsable --dependency="afterany:${USF_JOB%%;*}:${HCS_JOB%%;*}" \
  hpc/analyze_usf_encounter.slurm "$TAG")
cat <<MSG
tag            $TAG
USF manifest   $USF_MANIFEST ($USF_ROWS rows on $USF_WORKERS strided workers)  job ${USF_JOB%%;*}
HCS manifest   $HCS_MANIFEST ($HCS_ROWS rows, <=${HCS_CORES} concurrent)        job ${HCS_JOB%%;*}
analysis       job ${ANALYSIS_JOB%%;*} (afterany) -> results/usf_encounter_${TAG}/usf_vs_dem.{csv,png,summary.json}
                                               results/hcs_encounter_${TAG}/hcs_vs_dem.csv
MSG
