#!/bin/bash
# Run a DSMC_V2 Python command with a supported interpreter.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_PYTHONPATH="$ROOT/contracts/python:$ROOT/Coll_Models_v2/src:$ROOT/DSMC_0D_v2/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$PROJECT_PYTHONPATH:$PYTHONPATH"
else
    export PYTHONPATH="$PROJECT_PYTHONPATH"
fi

CANDIDATES=()
if [[ -n "${DSMC_V2_PYTHON:-}" ]]; then
    CANDIDATES+=("$DSMC_V2_PYTHON")
fi
CANDIDATES+=("$ROOT/.conda-v2/bin/python")
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    CANDIDATES+=("$VIRTUAL_ENV/bin/python")
fi
CANDIDATES+=("python3")

for PYTHON_CANDIDATE in "${CANDIDATES[@]}"; do
    if [[ "$PYTHON_CANDIDATE" == */* && ! -x "$PYTHON_CANDIDATE" ]]; then
        continue
    fi
    if "$PYTHON_CANDIDATE" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' \
        >/dev/null 2>&1; then
        exec "$PYTHON_CANDIDATE" "$@"
    fi
done

printf '%s\n' \
    'ERROR: DSMC_V2 requires Python 3.10 or newer.' \
    'The system python on Negishi may still be Python 3.6.' \
    'Create the supported project environment with:' \
    '  module load conda' \
    '  bash hpc/setup_negishi_env.sh' >&2
exit 2
