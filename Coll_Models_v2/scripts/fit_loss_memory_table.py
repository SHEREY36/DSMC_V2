#!/usr/bin/env python3
"""Fit the loss law's dependence on the incoming partition at every node.

For one CTC encounter let z = E_tr/E be the incoming translational share and
eps = 1 - E'/E the fraction of the pair energy lost.  At every
(alpha, theta, AR) node the conditional mean is fitted by least squares over
the raw CTC hits (the collision measure):

    E[eps | z] = c_t z + c_r (1 - z),

the loss rates of translational and rotational energy.  A smooth sphere loses
only translational energy (c_r = 0, eps proportional to z); long rods lose
both almost equally.  The runtime multiplies the encounter-scale loss draw by
s(z) = E[eps | z] / E[eps], so the node mean is unchanged and the loss follows
the pair's own energy split.  Writes the table read by
dsmc_v2.loss_memory.LossMemoryTable.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dsmc_v2_contracts.io import OI, _vec, load_run


def node_events(directories):
    parts = []
    for directory in directories:
        run = load_run(directory)
        ov = np.asarray(run.outcomes)["values"]
        m, inertia = float(run.metadata["mass"]), float(run.metadata["moi_perpendicular"])
        g0 = _vec(ov, OI, "c1_pre") - _vec(ov, OI, "c2_pre")
        g1 = _vec(ov, OI, "c1_post") - _vec(ov, OI, "c2_post")
        et0 = 0.25 * m * np.sum(g0 ** 2, 1)
        er0 = 0.5 * inertia * (np.sum(_vec(ov, OI, "omega1_pre") ** 2, 1)
                               + np.sum(_vec(ov, OI, "omega2_pre") ** 2, 1))
        e1 = 0.25 * m * np.sum(g1 ** 2, 1) + 0.5 * inertia * (
            np.sum(_vec(ov, OI, "omega1_post") ** 2, 1)
            + np.sum(_vec(ov, OI, "omega2_post") ** 2, 1))
        energy = et0 + er0
        parts.append(np.column_stack([et0 / energy, 1.0 - e1 / energy, energy]))
    return np.vstack(parts)


def fit_node(item):
    key, directories = item
    events = node_events(directories)
    z, eps, energy = events.T
    design = np.column_stack([z, 1.0 - z])
    (c_t, c_r), *_ = np.linalg.lstsq(design, eps, rcond=None)
    mean = float(eps.mean())
    edges = np.quantile(z, np.linspace(0.0, 1.0, 6))
    bins = np.clip(np.digitize(z, edges[1:-1]), 0, 4)
    quintile = [float(eps[bins == k].mean()) for k in range(5)]
    linear = [float(c_t * z[bins == k].mean() + c_r * (1.0 - z[bins == k].mean())) for k in range(5)]
    return key, {"c_t": float(c_t), "c_r": float(c_r), "eps_mean": mean,
                 "z_mean": float(z.mean()), "events": int(len(z)),
                 "quintile_eps": quintile, "quintile_linear": linear,
                 "energy_weighted_ratio": float(np.sum(eps * energy) / energy.sum() / mean),
                 "energy_weighted_ratio_linear": float(
                     np.sum((c_t * z + c_r * (1.0 - z)) * energy) / energy.sum() / mean)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
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
    with ProcessPoolExecutor(args.workers) as pool:
        for key, fitted in pool.map(fit_node, sorted(nodes.items())):
            if not (np.isfinite(fitted["c_t"]) and np.isfinite(fitted["c_r"])
                    and fitted["eps_mean"] >= 0.0):
                raise SystemExit(f"loss-memory fit failed at {key}: {fitted}")
            report.append({"alpha": key[0], "theta": key[1], "AR": key[2], **fitted})
            print(json.dumps({k: report[-1][k] for k in
                              ("alpha", "theta", "AR", "c_t", "c_r", "eps_mean", "events")}), flush=True)
    json.dump({"schema": "loss-memory-v1",
               "measure": "CTC encounters (raw hits), unweighted least squares",
               "law": "E[eps|z] = c_t z + c_r (1 - z); runtime scale s(z) = E[eps|z] / eps_mean",
               "nodes": report}, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
