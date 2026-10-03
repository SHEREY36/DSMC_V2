#!/usr/bin/env python3
"""Fit the energy-tilted angular-memory law at every (alpha, theta, AR) node.

Each node pools its CTC encounter shards (raw hits = collision measure) and
fits p(c | z, z') ~ exp[(eta1 + xi z' + rho1 z) c + (eta2 + zeta z' + rho2 z) P2(c)]
with weight E_f, the post-collision pair energy.  Writes the runtime JSON table
read by dsmc_v2.angular_memory.AngularMemoryTable.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from coll_models_v2.angular_memory_fit import fit
from dsmc_v2_contracts.io import OI, _vec, load_run

NAMES = ("eta1", "xi", "rho1", "eta2", "zeta", "rho2")


def node_events(directories):
    parts = []
    for directory in directories:
        run = load_run(directory)
        ov = np.asarray(run.outcomes)["values"]
        m, inertia = float(run.metadata["mass"]), float(run.metadata["moi_perpendicular"])
        g0 = _vec(ov, OI, "c1_pre") - _vec(ov, OI, "c2_pre")
        g1 = _vec(ov, OI, "c1_post") - _vec(ov, OI, "c2_post")
        et0 = 0.25 * m * np.sum(g0 ** 2, 1)
        et1 = 0.25 * m * np.sum(g1 ** 2, 1)
        er0 = 0.5 * inertia * (np.sum(_vec(ov, OI, "omega1_pre") ** 2, 1)
                               + np.sum(_vec(ov, OI, "omega2_pre") ** 2, 1))
        er1 = 0.5 * inertia * (np.sum(_vec(ov, OI, "omega1_post") ** 2, 1)
                               + np.sum(_vec(ov, OI, "omega2_post") ** 2, 1))
        cosine = np.einsum("ni,ni->n", g0, g1) / np.sqrt(
            np.sum(g0 ** 2, 1) * np.sum(g1 ** 2, 1))
        parts.append(np.column_stack([np.clip(cosine, -1, 1), et0 / (et0 + er0),
                                      et1 / (et1 + er1), et1 + er1]))
    return np.vstack(parts)


def fit_node(item):
    key, directories = item
    events = node_events(directories)
    model = fit(events[:, 0], events[:, 1], events[:, 2], weight=events[:, 3], form="memory")
    theta = model["theta"]
    ones = np.ones(len(events))
    a = np.column_stack([ones, events[:, 2], events[:, 1]]) @ theta[:3]
    b = np.column_stack([ones, events[:, 2], events[:, 1]]) @ theta[3:]
    support = [float(v) for v in (*np.percentile(a, [0.5, 99.5]), *np.percentile(b, [0.5, 99.5]))]
    return key, theta.tolist(), model["grad"], len(events), support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--base-table", default=None,
                        help="start from this table's nodes; refitted nodes replace them")
    parser.add_argument("shard_globs", nargs="+")
    args = parser.parse_args()
    nodes = {}
    for pattern in args.shard_globs:
        for directory in sorted(glob.glob(pattern)):
            if not (os.path.isfile(os.path.join(directory, "metadata_v2.json"))
                    and os.path.isfile(os.path.join(directory, "outcomes_v2.bin"))):
                continue
            md = json.load(open(os.path.join(directory, "metadata_v2.json")))
            if md.get("sampling_mode", "").startswith("dsmc_post_ntc"):
                continue
            key = (round(float(md["alpha"]), 6), round(float(md["theta"]), 6),
                   round(float(md["aspect_ratio"]), 6))
            nodes.setdefault(key, []).append(directory)
    report = []
    if args.base_table:
        refit = {(round(k[0], 4), round(k[1], 5), round(k[2], 4)) for k in nodes}
        for node in json.load(open(args.base_table))["nodes"]:
            key = (round(node["alpha"], 4), round(node["theta"], 5), round(node["AR"], 4))
            if key not in refit:
                report.append(node)
    with ProcessPoolExecutor(args.workers) as pool:
        for key, theta, grad, count, support in pool.map(fit_node, sorted(nodes.items())):
            if not (np.all(np.isfinite(theta)) and grad < 1.0e-8):
                raise SystemExit(f"angular-memory fit failed at {key}: gradient {grad}")
            report.append({"alpha": key[0], "theta": key[1], "AR": key[2],
                           "gradient_max": grad, "events": count,
                           "a_support": support[:2], "b_support": support[2:],
                           **dict(zip(NAMES, theta))})
            print(json.dumps(report[-1]), flush=True)
    json.dump({"schema": "angular-memory-v1", "parameter_names": list(NAMES),
               "measure": "CTC encounters, weight = post-collision pair energy E_f",
               "law": "p(c|z,z') ~ exp[(eta1+xi z'+rho1 z) c + (eta2+zeta z'+rho2 z) P2(c)]",
               "nodes": report}, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
