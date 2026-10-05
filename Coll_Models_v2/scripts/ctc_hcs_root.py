#!/usr/bin/env python3
"""Exact-CTC temperature ratio of the homogeneous cooling state.

For each (alpha, AR) plane and each theta node shard, the HCS drift of
ln theta per collision is evaluated from the exact CTC encounters (raw hits =
the collision measure of a Maxwellian gas at T_tr = theta, T_rot = 1):

    d ln theta / d tau = [ (2/3) <dE_tr> / T_tr - <dE_rot> / T_rot ] / (2 <k>),

with dE_tr = E' z' - E z and dE_rot = E'(1 - z') - E(1 - z) the modal energy
changes of one encounter and <k> its mean number of collisions (tau counts
collisions per particle; one encounter involves two particles).  The steady
ratio theta* is the zero of the drift, interpolated linearly in ln theta
between the bracketing nodes.  Its standard error comes from a bootstrap over
the 128 attempt blocks of every shard.

This is binary physics with a two-temperature Maxwellian incoming state: the
DSMC should reproduce it up to non-Gaussian effects and interpolation, the DEM
up to finite-density (many-body, correlation) effects.

    hpc/python.sh Coll_Models_v2/scripts/ctc_hcs_root.py --output roots.json \\
        'results/ctc/nodes/training/*/*'
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dsmc_v2_contracts.io import OI, _vec, contact_model_id, load_run


def node_blocks(directory):
    """Per attempt block (128): sums of dE_tr, dE_rot, k and the event count."""
    run = load_run(directory)
    out = np.asarray(run.outcomes)
    ov = out["values"]
    m, inertia = float(run.metadata["mass"]), float(run.metadata["moi_perpendicular"])
    g0 = _vec(ov, OI, "c1_pre") - _vec(ov, OI, "c2_pre")
    g1 = _vec(ov, OI, "c1_post") - _vec(ov, OI, "c2_post")
    et0 = 0.25 * m * np.sum(g0 ** 2, 1)
    et1 = 0.25 * m * np.sum(g1 ** 2, 1)
    er0 = 0.5 * inertia * (np.sum(_vec(ov, OI, "omega1_pre") ** 2, 1)
                           + np.sum(_vec(ov, OI, "omega2_pre") ** 2, 1))
    er1 = 0.5 * inertia * (np.sum(_vec(ov, OI, "omega1_post") ** 2, 1)
                           + np.sum(_vec(ov, OI, "omega2_post") ** 2, 1))
    block = out["block_id"].astype(int)
    sums = np.zeros((128, 4))
    for col, values in enumerate((et1 - et0, er1 - er0, out["n_contact"].astype(float),
                                  np.ones(len(out)))):
        sums[:, col] = np.bincount(block, weights=values, minlength=128)
    md = run.metadata
    t_tr = float(md["temperature_translational"])
    t_rot = float(md["temperature_rotational"])
    return {"alpha": float(md["alpha"]), "theta": t_tr / t_rot, "AR": float(md["aspect_ratio"]),
            "t_tr": t_tr, "t_rot": t_rot, "blocks": sums, "model": contact_model_id(md)}


def drift(node, weights=None):
    s = node["blocks"] if weights is None else node["blocks"] * weights[:, None]
    det, der, k, n = s.sum(0)
    return ((2.0 / 3.0) * (det / n) / node["t_tr"] - (der / n) / node["t_rot"]) / (2.0 * k / n)


def root(thetas, drifts):
    """Zero of the drift in ln theta (the stable one: drift falls through zero)."""
    lt = np.log(thetas)
    for i in range(len(lt) - 1):
        if drifts[i] > 0.0 >= drifts[i + 1]:
            f = drifts[i] / (drifts[i] - drifts[i + 1])
            return float(np.exp(lt[i] + f * (lt[i + 1] - lt[i]))), (float(thetas[i]), float(thetas[i + 1]))
    return float("nan"), None


def plane(item):
    (alpha, ar), nodes, n_boot, seed = item
    nodes = sorted(nodes, key=lambda n: n["theta"])
    thetas = np.array([n["theta"] for n in nodes])
    drifts = np.array([drift(n) for n in nodes])
    theta_star, bracket = root(thetas, drifts)
    rng = np.random.default_rng(seed)
    boots = []
    if np.isfinite(theta_star):
        for _ in range(n_boot):
            d = [drift(n, np.bincount(rng.integers(0, 128, 128), minlength=128).astype(float))
                 for n in nodes]
            boots.append(root(thetas, np.array(d))[0])
    boots = np.array(boots, float)
    good = boots[np.isfinite(boots)]
    return {"alpha": alpha, "AR": ar, "theta_star": theta_star,
            "theta_star_se": float(good.std(ddof=1)) if len(good) > 2 else None,
            "bracket": bracket, "theta_nodes": thetas.tolist(), "drift": drifts.tolist(),
            "bootstrap_failures": int(len(boots) - len(good)),
            "contact_model_id": nodes[0]["model"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--alpha", type=float, nargs="*", help="restrict to these restitutions")
    parser.add_argument("--AR", type=float, nargs="*", help="restrict to these aspect ratios")
    parser.add_argument("shard_globs", nargs="+")
    args = parser.parse_args()
    dirs = sorted({d for pattern in args.shard_globs for d in glob.glob(pattern)
                   if os.path.isfile(os.path.join(d, "_SUCCESS"))})
    keep = []
    for d in dirs:
        md = json.load(open(os.path.join(d, "metadata_v2.json")))
        if md.get("sampling_mode", "").startswith("dsmc_post_ntc") or float(md["alpha"]) >= 1.0:
            continue
        if args.alpha and not any(abs(float(md["alpha"]) - a) < 1e-9 for a in args.alpha):
            continue
        if args.AR and not any(abs(float(md["aspect_ratio"]) - a) < 1e-9 for a in args.AR):
            continue
        keep.append(d)
    with ProcessPoolExecutor(args.workers) as pool:
        nodes = list(pool.map(node_blocks, keep))
    models = {n["model"] for n in nodes}
    if len(models) > 1:
        raise SystemExit(f"shards of several contact models: {sorted(models)}")
    planes = {}
    for n in nodes:
        planes.setdefault((round(n["alpha"], 4), round(n["AR"], 4)), []).append(n)
    items = [(key, value, args.bootstrap, 1000 + i) for i, (key, value) in enumerate(sorted(planes.items()))]
    with ProcessPoolExecutor(args.workers) as pool:
        result = list(pool.map(plane, items))
    for r in result:
        se = f"{r['theta_star_se']:.4f}" if r["theta_star_se"] else "  n/a "
        print(f"alpha={r['alpha']:.2f} AR={r['AR']:.2f}  theta*_CTC={r['theta_star']:.4f} +- {se}  "
              f"bracket={r['bracket']}")
    json.dump({"schema": "ctc-hcs-root-v1", "contact_model_id": models.pop() if models else None,
               "law": "d ln theta/d tau = [(2/3)<dE_tr>/T_tr - <dE_rot>/T_rot]/(2<k>), zero in ln theta",
               "planes": result}, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
