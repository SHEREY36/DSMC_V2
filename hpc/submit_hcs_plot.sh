#!/bin/bash
# Rebuild only the HCS summary and figure from completed array outputs.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MANIFEST=${1:-manifests/hcs_validation.csv}

if ! hpc/python.sh -c 'import matplotlib' >/dev/null 2>&1; then
    printf '%s\n' \
        'ERROR: matplotlib is missing from the DSMC_V2 Python environment.' \
        'Repair the project environment once, then rerun this command:' \
        '  module load conda' \
        '  bash hpc/setup_negishi_env.sh' >&2
    exit 2
fi

mkdir -p logs results/hcs_validation
JOB_RAW=$(sbatch --parsable hpc/plot_hcs_validation.slurm "$MANIFEST")
JOB=${JOB_RAW%%;*}
echo "hcs_plot_job=$JOB (uses existing HCS outputs; does not rerun DSMC)"
echo "After completion inspect results/hcs_validation/{summary.json,hcs_theta_attraction.png}"
