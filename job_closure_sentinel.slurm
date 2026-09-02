#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -J ctc_sentinel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH -o logs/closure_sentinel_submit_%j.out
#SBATCH -e logs/closure_sentinel_submit_%j.err

set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
mkdir -p manifests logs
hpc/python.sh hpc/make_closure_manifest.py --stage sentinel \
    --samples 5000 --output manifests/closure_sentinel.csv
bash hpc/submit_manifest.sh manifests/closure_sentinel.csv
