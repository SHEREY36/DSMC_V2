#!/bin/bash
# One exact CTC replay of USF incoming pairs at any alpha (production gate).
# The replay alpha may differ from the alpha the pairs were harvested at: the
# gate compares the model with CTC on identical pairs, and the pair states are
# attached to the metadata for the model evaluation.
set -euo pipefail
MANIFEST=${1:?manifest required}
INDEX=${2:?row index required}
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"
ROW=$(awk -F, -v target="$INDEX" 'NR==target+2 {print; exit}' "$MANIFEST")
[[ -n "$ROW" ]] || { echo "manifest row $INDEX is missing" >&2; exit 2; }
IFS=, read -r TASK ALPHA THETA AR SOURCE_ALPHA SEED NSAMPLES SOURCE_JSON REPLAY OUT <<< "${ROW%$'\r'}"
[[ "$TASK" == "$INDEX" ]] || { echo "manifest task/index mismatch" >&2; exit 2; }
[[ -f "$OUT/_SUCCESS" ]] && { echo "complete: $OUT"; exit 0; }
rm -rf "$OUT"; mkdir -p "$(dirname "$OUT")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_PROC_BIND=spread OMP_PLACES=cores
(cd HS_CTC_v2 && ./build/SphCyl "$ALPHA" "$THETA" 1.0 "$AR" "$ROOT/$OUT" "$SEED" "$NSAMPLES" v2 1 "$ROOT/$REPLAY")
hpc/python.sh - "$OUT" "$SOURCE_JSON" <<'PY'
import json, sys
out, src = sys.argv[1], sys.argv[2]
meta = json.load(open(out + "/metadata_v2.json")); source = json.load(open(src))
meta.update(replay_source_cell_features=source["cell_features"],
            replay_source_domain_features=source["domain_features"],
            replay_source_alpha=source["alpha"], replay_source_metadata=src)
json.dump(meta, open(out + "/metadata_v2.json", "w"), indent=1)
PY
PYTHONPATH="$ROOT/contracts/python" hpc/python.sh HS_CTC_v2/scripts/finalize_run.py "$OUT"
