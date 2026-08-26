#!/bin/bash
# Create the isolated Python environment used by login-node tools and Slurm jobs.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${1:-"$ROOT/.conda-v2"}

if ! command -v conda >/dev/null 2>&1; then
    if type module >/dev/null 2>&1; then
        module load conda
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    printf '%s\n' \
        'ERROR: conda is not available.' \
        'On Negishi run `module load conda`, then rerun this script.' >&2
    exit 2
fi

if [[ -e "$ENV_DIR" && ! -x "$ENV_DIR/bin/python" ]]; then
    printf 'ERROR: %s exists but is not a usable conda environment.\n' "$ENV_DIR" >&2
    printf 'Choose another path or move that directory aside.\n' >&2
    exit 2
fi

CONDA_PACKAGES=(
    python=3.11
    pip
    'setuptools>=64'
    wheel
    'numpy>=1.23'
    'scipy>=1.9'
    'scikit-learn>=1.2'
    'pyyaml>=6'
)
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    conda create --prefix "$ENV_DIR" -y "${CONDA_PACKAGES[@]}"
else
    # This also repairs an environment left incomplete by an interrupted setup.
    conda install --prefix "$ENV_DIR" -y "${CONDA_PACKAGES[@]}"
fi

if ! "$ENV_DIR/bin/python" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    printf 'ERROR: the environment at %s does not provide Python 3.10+.\n' "$ENV_DIR" >&2
    exit 2
fi

"$ENV_DIR/bin/python" -m pip install --no-build-isolation --no-deps \
    -e "$ROOT/contracts" \
    -e "$ROOT/Coll_Models_v2" \
    -e "$ROOT/DSMC_0D_v2"

"$ENV_DIR/bin/python" -c \
    'import numpy, scipy, sklearn, yaml, dsmc_v2_contracts, coll_models_v2, dsmc_v2; print("DSMC_V2 Python environment ready")'
"$ENV_DIR/bin/python" --version
