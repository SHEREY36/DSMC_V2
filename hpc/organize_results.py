#!/usr/bin/env python3
"""Gather the campaign data into one layout, grouped by what each folder is for.

    results/
      ctc/nodes/training/<set>/   Maxwellian CTC node shards the closure is fitted on
      ctc/nodes/holdout/<set>/    independent CTC holdout shards (never fitted)
      ctc/nodes/test/<set>/       local pipeline tests
      ctc/nodes/v1/<set>/         first-generation CTC campaign
      ctc/replays/<set>/          exact CTC replays of DSMC-USF incoming pairs
      dsmc_harvest/<campaign>/    DSMC-USF collision-flux pair reservoirs (replay sources)
      closure_estimates/          node fits, QA reports, precompute payloads (unchanged)
      validation/usf/<campaign>/  DSMC USF runs and their analyses
      validation/hcs/<campaign>/  DSMC HCS runs and their analyses
      validation/production_gate/<tag>/   production-measure test summaries
      validation/dem_comparison/<name>/   DEM-DSMC comparison reports and figures
      archive/<name>/             dry runs and smoke tests

Every move is a rename on the same file system, so nothing is copied.  Shard
folder names never change, so node estimates stay current (the QA compares
shard names, not full paths).  Paths inside manifests/*.csv are rewritten, and
symbolic links that pointed into a moved tree are retargeted.  The old -> new
map is appended to results/path_map.json; a second run finds nothing to do.

    hpc/python.sh hpc/organize_results.py            # show the plan
    hpc/python.sh hpc/organize_results.py --apply    # carry it out
"""
import argparse
import json
import os
import re
from pathlib import Path

EXPLICIT = [
    ("results/ctc/pilot", "results/ctc/nodes/v1/pilot"),
    ("results/ctc/production", "results/ctc/nodes/v1/production"),
    ("results/ctc_closure/sentinel", "results/ctc/nodes/training/sentinel_early"),
    ("results/ctc_closure_200k/sentinel", "results/ctc/nodes/training/sentinel"),
    ("results/ctc_closure_200k/ar_extension", "results/ctc/nodes/training/ar_extension"),
    ("results/ctc_closure_200k/ar_near_sphere", "results/ctc/nodes/training/ar_near_sphere"),
    ("results/ctc_closure_200k/ar_low_theta", "results/ctc/nodes/training/ar_low_theta"),
    ("results/ctc_closure_200k/alpha_refinement", "results/ctc/nodes/training/alpha_refinement"),
    ("results/ctc_ar_ge2", "results/ctc/nodes/training/ar_ge2"),
    ("results/ctc_independent_holdout_v1", "results/ctc/nodes/holdout/independent_holdout_v1"),
    ("results/ctc_alpha_slice_test", "results/ctc/nodes/test/alpha_slice_test"),
    ("results/ctc_hcs_probe_local", "results/ctc/nodes/test/hcs_probe_local"),
    ("outputs/usf_nonlinear_v1_20260928_ctc/replay", "results/ctc/replays/usf_nonlinear_v1_20260928"),
    ("outputs/usf_nonlinear_v2_20260928_ctc/replay", "results/ctc/replays/usf_nonlinear_v2_20260928"),
    ("results/usf_nonlinear_v1_20260928/harvest", "results/dsmc_harvest/usf_nonlinear_v1_20260928"),
    ("results/usf_nonlinear_v2_20260928/harvest", "results/dsmc_harvest/usf_nonlinear_v2_20260928"),
]
ARCHIVE = re.compile(r"^(contract_.*|local_smoke)$")
KEEP = {"ctc", "dsmc_harvest", "closure_estimates", "validation", "archive"}


def plan(root: Path) -> list[tuple[str, str]]:
    moves = [(old, new) for old, new in EXPLICIT if (root / old).exists()]
    moved = {old for old, _ in moves}
    for entry in sorted((root / "results").iterdir()):
        name = entry.name
        rel = f"results/{name}"
        if name in KEEP or rel in moved or name == "path_map.json":
            continue
        if entry.is_file():
            # loose campaign records (submission logs, retry ledgers) go with their kind
            kind = "usf" if name.startswith("usf_") else "hcs" if name.startswith("hcs_") else None
            if kind:
                moves.append((rel, f"results/validation/{kind}/records/{name}"))
            continue
        gate = re.match(r"^production_gate_(.+)$", name)
        if gate:
            if (entry / "replay").is_dir():
                moves.append((f"{rel}/replay", f"results/ctc/replays/gate_{gate.group(1)}"))
            moves.append((rel, f"results/validation/production_gate/{gate.group(1)}"))
        elif name.startswith("usf_dem_dsmc_comparison_"):
            moves.append((rel, f"results/validation/dem_comparison/{name}"))
        elif name.startswith("usf_"):
            moves.append((rel, f"results/validation/usf/{name}"))
        elif name.startswith("hcs_"):
            moves.append((rel, f"results/validation/hcs/{name}"))
        elif ARCHIVE.match(name):
            moves.append((rel, f"results/archive/{name}"))
    return moves


def symlinks_under(root: Path, sources: list[str]) -> list[tuple[Path, Path]]:
    links = []
    for source in sources:
        base = root / source
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            for name in dirnames + filenames:
                path = Path(dirpath) / name
                if path.is_symlink():
                    links.append((path, Path(os.path.normpath(path.parent / os.readlink(path)))))
    return links


def remap(path: Path, root: Path, moves: list[tuple[str, str]]) -> Path:
    rel = os.path.relpath(path, root)
    for old, new in sorted(moves, key=lambda m: -len(m[0])):
        if rel == old or rel.startswith(old + "/"):
            return root / (new + rel[len(old):])
    return path


def rewrite_text(text: str, moves: list[tuple[str, str]]) -> str:
    for old, new in sorted(moves, key=lambda m: -len(m[0])):
        text = re.sub(re.escape(old) + r"(?=[/,\"'\s]|$)", new, text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    moves = plan(root)
    for old, new in moves:
        if (root / new).exists():
            raise SystemExit(f"target already exists, refusing to merge: {new}")
    print(f"{len(moves)} moves" + ("" if args.apply else " (dry run; pass --apply to carry out)"))
    for old, new in moves:
        print(f"  {old:62s} -> {new}")
    manifests = sorted((root / "manifests").glob("*.csv")) if (root / "manifests").is_dir() else []
    changed = [m for m in manifests if rewrite_text(m.read_text(), moves) != m.read_text()]
    print(f"{len(changed)} manifests carry paths that will be rewritten")
    links = symlinks_under(root, [old for old, _ in moves])
    print(f"{len(links)} symbolic links inside moved folders will be retargeted")
    if not args.apply or not moves:
        return

    targets = [(link, target) for link, target in links]
    for old, new in moves:
        destination = root / new
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(root / old, destination)
    for link, target in targets:
        new_link = remap(link, root, moves)
        new_target = remap(target, root, moves)
        os.remove(new_link)
        os.symlink(os.path.relpath(new_target, new_link.parent), new_link)
    for manifest in changed:
        manifest.write_text(rewrite_text(manifest.read_text(), moves))
    for old, _ in moves:
        parent = (root / old).parent
        while parent != root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    record = root / "results" / "path_map.json"
    history = json.loads(record.read_text()) if record.is_file() else []
    history.append({"moves": moves})
    record.write_text(json.dumps(history, indent=1) + "\n")
    print(f"done; map appended to {record.relative_to(root)}")


if __name__ == "__main__":
    main()
