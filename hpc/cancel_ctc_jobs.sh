#!/bin/bash
# Snapshot and cancel only CTC Slurm array parents; preserve every run byte.
set -euo pipefail

MODE=${1:-}
if [[ "$MODE" != "--cancel" ]]; then
    printf 'Usage: bash hpc/cancel_ctc_jobs.sh --cancel\n' >&2
    printf 'The required flag prevents an accidental cancellation.\n' >&2
    exit 2
fi
if ! command -v squeue >/dev/null 2>&1 || ! command -v sacct >/dev/null 2>&1 \
        || ! command -v scancel >/dev/null 2>&1; then
    printf 'ERROR: run this script on a Slurm login node (squeue/sacct/scancel required).\n' >&2
    exit 2
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SNAPSHOT="$ROOT/reports/operations/ctc_cancel_$STAMP"
mkdir -p "$SNAPSHOT"
git -C "$ROOT" rev-parse HEAD > "$SNAPSHOT/git_sha.txt"
squeue -u "$USER" -o '%.18i|%.18A|%.9P|%.60j|%.2t|%.10M|%R' \
    > "$SNAPSHOT/squeue_before.txt"
sacct -u "$USER" --starttime 2026-08-01 \
    --format=JobIDRaw,JobName%60,State,Submit,Start,End,Elapsed,NNodes,NCPUS -n -P \
    > "$SNAPSHOT/sacct_before.txt"
find "$ROOT/results" -path "$ROOT/results/quarantine" -prune -o \
    -type f -name _SUCCESS -print 2>/dev/null \
    > "$SNAPSHOT/completed_success_markers.txt" || true
awk 'END {print NR + 0}' "$SNAPSHOT/completed_success_markers.txt" \
    > "$SNAPSHOT/completed_node_count.txt"
find "$ROOT/results" -path "$ROOT/results/quarantine" -prune -o \
    -type f -name metadata_v2.json -print 2>/dev/null \
    > "$SNAPSHOT/all_run_metadata.txt" || true
comm -23 \
    <(sed 's|/metadata_v2.json$||' "$SNAPSHOT/all_run_metadata.txt" | sort) \
    <(sed 's|/_SUCCESS$||' "$SNAPSHOT/completed_success_markers.txt" | sort) \
    > "$SNAPSHOT/incomplete_directories_quarantined.txt"
if [[ -d "$ROOT/manifests" ]]; then
    cp -a "$ROOT/manifests" "$SNAPSHOT/manifests"
fi

# %A collapses array tasks to parents. The requested scope is every job whose
# name begins with CTC, including historical sweep names.
mapfile -t JOBS < <(squeue -h -u "$USER" -o '%A|%j' \
    | awk -F'|' 'tolower($2) ~ /^ctc([_-]|$)/ {print $1}' \
    | sort -u)
: > "$SNAPSHOT/cancelled_parent_job_ids.txt"
if (( ${#JOBS[@]} )); then
    printf '%s\n' "${JOBS[@]}" > "$SNAPSHOT/cancelled_parent_job_ids.txt"
    scancel "${JOBS[@]}"
    JOB_CSV=$(IFS=,; printf '%s' "${JOBS[*]}")
    for _ in {1..30}; do
        [[ -z "$(squeue -h -j "$JOB_CSV")" ]] && break
        sleep 2
    done
    if [[ -n "$(squeue -h -j "$JOB_CSV")" ]]; then
        printf 'ERROR: cancelled jobs are still present; incomplete runs were not moved.\n' >&2
        exit 3
    fi
fi
squeue -u "$USER" -o '%.18i|%.18A|%.9P|%.60j|%.2t|%.10M|%R' \
    > "$SNAPSHOT/squeue_after.txt"

# Move incomplete directories after cancellation, retaining their full
# contents while guaranteeing that fresh runs cannot overwrite them.
QUARANTINE="$ROOT/results/quarantine/ctc_cancel_$STAMP"
mkdir -p "$QUARANTINE"
: > "$SNAPSHOT/quarantine_map.txt"
while IFS= read -r SOURCE; do
    [[ -d "$SOURCE" ]] || continue
    RELATIVE=${SOURCE#"$ROOT/results/"}
    DESTINATION="$QUARANTINE/$RELATIVE"
    mkdir -p "$(dirname "$DESTINATION")"
    mv -- "$SOURCE" "$DESTINATION"
    printf '%s|%s\n' "$SOURCE" "$DESTINATION" >> "$SNAPSHOT/quarantine_map.txt"
done < "$SNAPSHOT/incomplete_directories_quarantined.txt"
awk 'END {print NR + 0}' "$SNAPSHOT/quarantine_map.txt" \
    > "$SNAPSHOT/quarantined_directory_count.txt"
printf 'Snapshot: %s\nCancelled CTC array parents: %d\nQuarantine: %s\n' \
    "$SNAPSHOT" "${#JOBS[@]}" "$QUARANTINE"
