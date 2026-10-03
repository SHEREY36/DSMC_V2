#!/usr/bin/env python3
"""Carry the kinematic-propensity cache across the results/ reorganisation.

The cache used to be keyed by a shard's full resolved path, so moving the
shards (hpc/organize_results.py) would recompute every propensity, the most
expensive part of a node fit.  The key is now the shard name, seed and attempt
file size and mtime (coll_models_v2.estimate.propensity_identity).  This
renames every cache file written under an old path to the new key, using
results/path_map.json to recover the old path.  Safe to run more than once.

    hpc/python.sh hpc/migrate_propensity_cache.py
"""
import hashlib
import json
import os
from pathlib import Path

from coll_models_v2.estimate import PROPENSITY_CACHE, propensity_identity


def old_paths(relative: str, history: list) -> list[str]:
    """Every earlier location of a path, newest move first."""
    found, current = [], relative
    for record in reversed(history):
        for old, new in sorted(record["moves"], key=lambda m: -len(m[1])):
            if current == new or current.startswith(new + "/"):
                current = old + current[len(new):]
                found.append(current)
                break
    return found


def main():
    root = Path.cwd().resolve()
    cache = root / PROPENSITY_CACHE
    record = root / "results" / "path_map.json"
    if not cache.is_dir() or not record.is_file():
        print("nothing to migrate"); return
    history = json.loads(record.read_text())
    files = {f.name for f in cache.iterdir()}
    renamed = present = 0
    for attempts in (root / "results" / "ctc").rglob("attempts_v2.bin"):
        shard = attempts.parent
        meta = json.loads((shard / "metadata_v2.json").read_text())
        stat = attempts.stat()
        new_id = propensity_identity(shard.resolve(), meta.get("seed"), stat)
        candidates = [shard.resolve()] + [root / p for p in old_paths(
            str(shard.resolve().relative_to(root)), history)]
        for path in candidates:
            old_id = hashlib.sha256((f"{path}|{meta.get('seed')}|{stat.st_size}|"
                                     f"{stat.st_mtime_ns}").encode()).hexdigest()[:16]
            prefix = f"{shard.name}_{old_id}_{stat.st_size}_"
            for name in [f for f in files if f.startswith(prefix) and f.endswith(".npy")]:
                target = f"{shard.name}_{new_id}_{stat.st_size}_{name[len(prefix):]}"
                if target in files:
                    present += 1
                    continue
                os.rename(cache / name, cache / target)
                files.discard(name); files.add(target)
                renamed += 1
    print(f"propensity cache: {renamed} files re-keyed, {present} already current, "
          f"{len(files)} files in {cache.relative_to(root)}")


if __name__ == "__main__":
    main()
