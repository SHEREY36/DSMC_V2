#!/usr/bin/env python3
"""Production gate: collisional productions of the model versus exact CTC.

Each input is an exact CTC replay of USF incoming pairs (post-NTC collision-flux
samples) at restitution alpha.  On those identical pairs the production model --
encounter clock, per-encounter loss, energy kernel and angular-memory law at
(alpha, theta, AR) -- is compared with CTC through the rates that fix the USF
steady state (moment balance, report Section 1):

  rot_residual  (Lambda_rot^model - Lambda_rot^CTC) / D^CTC   sets theta = T_tr/T_rot
  theta_shift   -kappa * rot_residual, the predicted relative theta error
  loss_ratio    D^model / D^CTC                               sets T*
  xy_ratio, N1_ratio  traceless-stress production ratios      set P*_xy, N1

theta is the root of Lambda_rot = 0, so a residual moves it by
dtheta/theta = -D / (theta dLambda_rot/dtheta) * rot_residual = -kappa rot_residual.
To leading order in the loss, kappa = 2 eps_E (1 + theta) / (theta p_exch), with
eps_E the energy-weighted loss per encounter (from CTC) and p_exch = 1 - b the
exchange fraction of the energy kernel.  D -> 0 as alpha -> 1, so rot_residual
and its error grow like 1/eps_E there while kappa falls like eps_E; the rotational
check is therefore made on theta_shift, which stays on the scale of the observable.

Per flux sample the CTC production is (A_prop/N_att) sum_hits dX and the model
production is sigma(x) E[dX | x]; the common n^2 <g>/2 cancels.  Standard errors
come from contiguous batch means on both sides.  A replay passes when every
metric is within its tolerance or within three standard errors.
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dsmc_v2.angular_memory import AngularMemoryTable, direction_from_cosine, sample_cosine
from dsmc_v2.artifact import FEATURE_NAMES, VariationalClosure
from dsmc_v2.encounter import EncounterClock, mean_projected_area
from dsmc_v2.legacy_models import FrozenLossModel
from dsmc_v2.loss_memory import LossMemoryTable
from dsmc_v2_contracts.io import AI, OI, _vec as vec, load_run

LOSS_CAP = 1.0 - 1.0e-9
TOLERANCE = {"theta_shift": 0.02, "loss_ratio": 0.03, "xy_ratio": 0.04}
ARGS = None
_CACHE = {}


def objects():
    if not _CACHE:
        closure = VariationalClosure(ARGS.artifact)
        memory = AngularMemoryTable(ARGS.angular_memory)
        memory.bind(closure.coordinates)
        loss_memory = None
        if getattr(ARGS, "loss_memory", None):
            loss_memory = LossMemoryTable(ARGS.loss_memory)
            loss_memory.bind(closure.coordinates)
        _CACHE.update(closure=closure, memory=memory, clock=EncounterClock(ARGS.encounter_table),
                      loss=FrozenLossModel(ARGS.model_root, 1.21, 3.67), loss_memory=loss_memory)
    return _CACHE


def a_perp(u1, u2, gh, ar):
    length = ar - 1.0
    s1 = np.linalg.norm(np.cross(u1, gh), axis=1)
    s2 = np.linalg.norm(np.cross(u2, gh), axis=1)
    triple = np.abs(np.einsum("ni,ni->n", gh, np.cross(u1, u2)))
    return np.pi + 2 * length * (s1 + s2) + length ** 2 * triple


def batch_mean_se(values, batches=40):
    values = np.asarray(values, float)
    n = len(values) // batches * batches
    means = values[:n].reshape(batches, -1).mean(1)
    return float(values.mean()), float(means.std(ddof=1) / np.sqrt(batches))


def evaluate(directory):
    o = objects(); cl = o["closure"]
    run = load_run(directory); md = run.metadata
    alpha, ar, theta = float(md["alpha"]), float(md["aspect_ratio"]), float(md["theta"])
    m, inertia = float(md["mass"]), float(md["moi_perpendicular"])
    att, out = np.asarray(run.attempts), np.asarray(run.outcomes)
    n_att = len(att)
    # CTC contributions per attempt (zero for misses) -> batch errors respect the flux sample
    key_a = att["event_id"] * 100000 + att["attempt_index"]
    key_o = out["event_id"] * 100000 + out["attempt_index"]
    order = np.argsort(key_a)
    pos = order[np.searchsorted(key_a[order], key_o)]
    ov = out["values"]
    g0 = vec(ov, OI, "c1_pre") - vec(ov, OI, "c2_pre"); g1 = vec(ov, OI, "c1_post") - vec(ov, OI, "c2_post")
    et0, et1 = .25 * m * np.sum(g0 ** 2, 1), .25 * m * np.sum(g1 ** 2, 1)
    er0 = .5 * inertia * (np.sum(vec(ov, OI, "omega1_pre") ** 2, 1) + np.sum(vec(ov, OI, "omega2_pre") ** 2, 1))
    er1 = .5 * inertia * (np.sum(vec(ov, OI, "omega1_post") ** 2, 1) + np.sum(vec(ov, OI, "omega2_post") ** 2, 1))
    scale = float(md["proposal_area"])
    C = np.zeros((n_att, 6))
    C[pos, 0] = et0 + er0 - et1 - er1
    C[pos, 1] = er1 - er0
    C[pos, 2] = g1[:, 0] * g1[:, 1] - g0[:, 0] * g0[:, 1]
    C[pos, 3] = (g1[:, 0] ** 2 - g1[:, 1] ** 2) - (g0[:, 0] ** 2 - g0[:, 1] ** 2)
    C[pos, 4] = 1.0
    C[pos, 5] = et0 + er0
    C *= scale
    # the production model on a random subset of the same flux sample
    rng = np.random.default_rng(ARGS.seed)
    idx = np.sort(rng.choice(n_att, size=min(ARGS.samples, n_att), replace=False))
    av = att["values"][idx]
    g = vec(av, AI, "c1") - vec(av, AI, "c2"); speed = np.linalg.norm(g, axis=1); gh = g / speed[:, None]
    o1, o2 = vec(av, AI, "omega1"), vec(av, AI, "omega2")
    state = cl.kernel_state(alpha, theta, ar)
    xi = (np.linalg.norm(o1, axis=1) + np.linalg.norm(o2, axis=1)) * ar / speed
    w = (o["clock"].sigma(theta, ar, 1.0) * a_perp(vec(av, AI, "u1"), vec(av, AI, "u2"), gh, ar)
         * np.interp(xi, cl.xi_grid, state["xi_enhancement"]) / mean_projected_area(ar, 1.0))
    lp = o["loss"].loss_parameters(alpha, ar) if alpha < 1.0 else {"gamma_max": 0.0, "one_hit_probability": 1.0}
    bl = 1.21 / 4.88 * lp["gamma_max"] * lp["one_hit_probability"]
    fitted = float(state["fitted_mean_loss"])
    loss_scale = fitted / bl if bl > 0 else 0.0
    et, er = .25 * m * speed ** 2, .5 * inertia * (np.sum(o1 ** 2, 1) + np.sum(o2 ** 2, 1))
    energy = et + er; z = et / energy
    eps = rng.beta(1.21, 3.67, size=len(z)) * lp["gamma_max"] * lp["one_hit_probability"] * loss_scale
    if o["loss_memory"] is not None:
        # the runtime's loss law: eps = E[eps|z] B/<B> with the table's node means
        rates = o["loss_memory"].stencil(state)
        eps = np.minimum(eps * (rates[2] / fitted) * o["loss_memory"].scale(rates, z), LOSS_CAP)
    stencil = o["memory"].stencil(state)
    zf = np.array([cl.sample_energy(state, z[i], eps[i], rng, loss_mean=fitted) for i in range(len(z))])
    gp = np.array([2 * np.sqrt(zf[i] * energy[i] * (1 - eps[i]) / m)
                   * direction_from_cosine(gh[i], sample_cosine(stencil, z[i], zf[i], rng), rng)
                   for i in range(len(z))])
    erf = (1 - zf) * energy * (1 - eps)
    D = np.column_stack([w * eps * energy, w * (erf - er),
                         w * (gp[:, 0] * gp[:, 1] - g[:, 0] * g[:, 1]),
                         w * ((gp[:, 0] ** 2 - gp[:, 1] ** 2) - (g[:, 0] ** 2 - g[:, 1] ** 2)), w])
    c = [batch_mean_se(C[:, j]) for j in range(5)]
    d = [batch_mean_se(D[:, j]) for j in range(5)]

    def ratio(j):
        value = d[j][0] / c[j][0]
        return value, abs(value) * float(np.hypot(d[j][1] / d[j][0], c[j][1] / c[j][0]))
    rot = (d[1][0] - c[1][0]) / c[0][0]
    rot_se = float(np.hypot(d[1][1], c[1][1]) / abs(c[0][0]))
    eps_energy = float(C[:, 0].sum() / C[:, 5].sum())
    p_exch = float(state.get("p_exch", np.nan))
    kappa = 2 * eps_energy * (1 + theta) / (theta * p_exch) if p_exch > 0 else float("nan")
    row = {"directory": Path(directory).name, "alpha": alpha, "AR": ar, "theta": theta,
           "source_alpha": md.get("replay_source_alpha"),
           "node_alpha": bool(np.any(np.isclose(cl.coordinates[:, 0], alpha, atol=1e-9))),
           "rot_residual": rot, "rot_residual_se": rot_se,
           "eps_energy": eps_energy, "p_exch": p_exch, "kappa": kappa,
           "theta_shift": -kappa * rot, "theta_shift_se": kappa * rot_se}
    for name, j in (("loss_ratio", 0), ("xy_ratio", 2), ("N1_ratio", 3), ("rate_ratio", 4)):
        row[name], row[name + "_se"] = ratio(j)
    checks = {"theta_shift": abs(row["theta_shift"]) <= max(TOLERANCE["theta_shift"], 3 * row["theta_shift_se"]),
              "loss_ratio": abs(row["loss_ratio"] - 1) <= max(TOLERANCE["loss_ratio"], 3 * row["loss_ratio_se"]),
              "xy_ratio": abs(row["xy_ratio"] - 1) <= max(TOLERANCE["xy_ratio"], 3 * row["xy_ratio_se"])}
    row["pass"] = bool(all(checks.values()))
    row["failed"] = ";".join(k for k, ok in checks.items() if not ok)
    return row


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--angular-memory", required=True)
    parser.add_argument("--encounter-table", required=True)
    parser.add_argument("--loss-memory", default=None,
                        help="loss-memory table; omit for a loss independent of z")
    parser.add_argument("--model-root", default="DSMC_0D_v2/models")
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, help="output prefix (.csv and .json are written)")
    parser.add_argument("replays", nargs="+")
    ARGS = parser.parse_args()
    replays = [r for r in ARGS.replays if Path(r, "_SUCCESS").is_file()]
    with ProcessPoolExecutor(ARGS.workers) as pool:
        rows = list(pool.map(evaluate, replays))
    rows.sort(key=lambda r: (r["AR"], r["alpha"], str(r["source_alpha"])))
    out = Path(ARGS.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {"replays": len(rows), "pass": all(r["pass"] for r in rows),
               "failed": [r["directory"] + ":" + r["failed"] for r in rows if not r["pass"]],
               "tolerance": TOLERANCE,
               "median_abs_rot_residual": {
                   "nodes": float(np.median([abs(r["rot_residual"]) for r in rows if r["node_alpha"]] or [np.nan])),
                   "between_nodes": float(np.median([abs(r["rot_residual"]) for r in rows if not r["node_alpha"]] or [np.nan]))},
               "median_abs_theta_shift": {
                   "nodes": float(np.median([abs(r["theta_shift"]) for r in rows if r["node_alpha"]] or [np.nan])),
                   "between_nodes": float(np.median([abs(r["theta_shift"]) for r in rows if not r["node_alpha"]] or [np.nan]))}}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    for r in rows:
        print(f"AR {r['AR']:.2f} alpha {r['alpha']:.3f} {'node' if r['node_alpha'] else 'interp'} "
              f"rot {r['rot_residual']:+.4f}+-{r['rot_residual_se']:.4f} "
              f"dtheta {100 * r['theta_shift']:+.2f}+-{100 * r['theta_shift_se']:.2f}% loss {r['loss_ratio']:.4f} "
              f"xy {r['xy_ratio']:.4f} N1 {r['N1_ratio']:.4f} {'PASS' if r['pass'] else 'FAIL ' + r['failed']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
