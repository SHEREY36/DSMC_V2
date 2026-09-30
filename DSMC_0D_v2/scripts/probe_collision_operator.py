#!/usr/bin/env python3
"""Compare DSMC collision operators with exact CTC on identical incoming pairs.

Input: CTC replay directories whose attempts are DSMC post-NTC collision-flux
samples (sampling_mode dsmc_post_ntc_replay_v1).  For any pair observable X the
production per flux sample is

  CTC  : (A_prop / n_attempts) * sum_{hit encounters} Delta X
  DSMC : sigma(x) * E[Delta X | x],  sigma(x) = sigma_c r A_perp(x) g(Xi)/A_mean

and the common factor n^2 <g>/2 cancels in the ratio.  Operators:
  V0 legacy          contact clock, BL per-contact loss, artifact angular law
  V1 encounter       encounter clock r(theta, AR), encounter-scale loss
  V2 encounter+memory  V1 + angular-memory law
Reported: DSMC/CTC ratios of event rate, dissipation, translational energy,
xy and N1 traceless-stress productions, and the rotational-production
residual in units of the CTC dissipation (zero means Lambda_rot is right).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dsmc_v2.angular_memory import AngularMemoryTable, direction_from_cosine, sample_cosine
from dsmc_v2.artifact import FEATURE_NAMES, VariationalClosure
from dsmc_v2.encounter import EncounterClock
from dsmc_v2.legacy_models import FrozenLossModel
from dsmc_v2_contracts.io import AI, OI, _vec as vec, load_run

ARGS = None
_CACHE = {}


def objects():
    if not _CACHE:
        closure = VariationalClosure(ARGS.artifact, corrections_enabled=False)
        memory = AngularMemoryTable(ARGS.angular_memory)
        memory.bind(closure.coordinates)
        _CACHE.update(closure=closure, memory=memory, clock=EncounterClock(ARGS.encounter_table),
                      loss=FrozenLossModel(ARGS.model_root, 1.21, 3.67))
    return _CACHE


def a_perp(u1, u2, gh, ar):
    length = ar - 1.0
    s1 = np.linalg.norm(np.cross(u1, gh), axis=1)
    s2 = np.linalg.norm(np.cross(u2, gh), axis=1)
    triple = np.abs(np.einsum("ni,ni->n", gh, np.cross(u1, u2)))
    return np.pi + 2 * length * (s1 + s2) + length ** 2 * triple


def a_mean(ar):
    rng = np.random.default_rng(0x5EED)
    u = rng.normal(size=(3, 200000, 3))
    u /= np.linalg.norm(u, axis=2, keepdims=True)
    return float(np.mean(a_perp(u[0], u[1], u[2], ar)))


def tl(g):
    return np.column_stack([g[:, 0] * g[:, 1], g[:, 0] ** 2 - g[:, 1] ** 2])


def probe(directory):
    o = objects(); cl = o["closure"]
    run = load_run(directory); md = run.metadata
    alpha, ar, theta = float(md["alpha"]), float(md["aspect_ratio"]), float(md["theta"])
    m, inertia = float(md["mass"]), float(md["moi_perpendicular"])
    att, out = np.asarray(run.attempts), np.asarray(run.outcomes)
    ov = out["values"]
    g0 = vec(ov, OI, "c1_pre") - vec(ov, OI, "c2_pre"); g1 = vec(ov, OI, "c1_post") - vec(ov, OI, "c2_post")
    et0, et1 = .25 * m * np.sum(g0 ** 2, 1), .25 * m * np.sum(g1 ** 2, 1)
    er0 = .5 * inertia * (np.sum(vec(ov, OI, "omega1_pre") ** 2, 1) + np.sum(vec(ov, OI, "omega2_pre") ** 2, 1))
    er1 = .5 * inertia * (np.sum(vec(ov, OI, "omega1_post") ** 2, 1) + np.sum(vec(ov, OI, "omega2_post") ** 2, 1))
    k = float(md["proposal_area"]) / len(att)
    ctc = dict(rate=k * len(out), loss=k * np.sum(et0 + er0 - et1 - er1), rot=k * np.sum(er1 - er0),
               tr=k * np.sum(et1 - et0), tl=k * np.sum(tl(g1) - tl(g0), 0))
    rng = np.random.default_rng(ARGS.seed)
    idx = np.sort(rng.choice(len(att), size=min(ARGS.samples, len(att)), replace=False))
    av = att["values"][idx]
    g = vec(av, AI, "c1") - vec(av, AI, "c2"); speed = np.linalg.norm(g, axis=1); gh = g / speed[:, None]
    o1, o2 = vec(av, AI, "omega1"), vec(av, AI, "omega2")
    feats = np.array([md["replay_source_cell_features"][q] for q in FEATURE_NAMES])
    state = cl.kernel_state(alpha, theta, ar, feats, feats)
    xi = (np.linalg.norm(o1, axis=1) + np.linalg.norm(o2, axis=1)) * ar / speed
    shape = (a_perp(vec(av, AI, "u1"), vec(av, AI, "u2"), gh, ar)
             * np.interp(xi, cl.xi_grid, state["xi_enhancement"]) / a_mean(ar))
    lp = o["loss"].loss_parameters(alpha, ar)
    bl = 1.21 / 4.88 * lp["gamma_max"] * lp["one_hit_probability"]
    fitted = float(state["fitted_mean_loss"])
    et, er = .25 * m * speed ** 2, .5 * inertia * (np.sum(o1 ** 2, 1) + np.sum(o2 ** 2, 1))
    energy = et + er; z = et / energy
    gamma = rng.beta(1.21, 3.67, size=len(z)) * lp["gamma_max"] * lp["one_hit_probability"]
    stencil = o["memory"].stencil(state)
    row = dict(directory=Path(directory).name, alpha=alpha, AR=ar, theta=theta)
    sigma_c = float(md["collision_cross_section"])
    for name, clock, scale, memory in (("V0", 1.0, 1.0, False),
                                       ("V1", o["clock"].ratio(theta, ar), fitted / bl, False),
                                       ("V2", o["clock"].ratio(theta, ar), fitted / bl, True)):
        w = sigma_c * clock * shape; eps = gamma * scale
        zf = np.array([cl.sample_energy(state, z[i], eps[i], rng, loss_mean=bl * scale) for i in range(len(z))])
        gp = np.empty_like(g)
        for i in range(len(z)):
            direction = (direction_from_cosine(gh[i], sample_cosine(stencil, z[i], zf[i], rng), rng)
                         if memory else cl.sample_direction(gh[i], state, zf[i], rng))
            gp[i] = 2 * np.sqrt(zf[i] * energy[i] * (1 - eps[i]) / m) * direction
        etf, erf = zf * energy * (1 - eps), (1 - zf) * energy * (1 - eps)
        dt = np.mean(w[:, None] * (tl(gp) - tl(g)), 0)
        row.update({f"{name}_rate": np.mean(w) / ctc["rate"],
                    f"{name}_loss": np.mean(w * eps * energy) / ctc["loss"],
                    f"{name}_tr": np.mean(w * (etf - et)) / ctc["tr"],
                    f"{name}_xy": dt[0] / ctc["tl"][0], f"{name}_N1": dt[1] / ctc["tl"][1],
                    f"{name}_rot_residual": (np.mean(w * (erf - er)) - ctc["rot"]) / ctc["loss"]})
    return row


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", default="models/microscopic_closure_v2_angular_evidence/closure_v2.npz")
    parser.add_argument("--encounter-table", default="DSMC_0D_v2/models/encounter_cross_section_v1.json")
    parser.add_argument("--angular-memory", default="DSMC_0D_v2/models/angular_memory_v1.json")
    parser.add_argument("--model-root", default="DSMC_0D_v2/models")
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("replays", nargs="+")
    ARGS = parser.parse_args()
    with ProcessPoolExecutor(ARGS.workers) as pool:
        rows = list(pool.map(probe, ARGS.replays))
    frame = pd.DataFrame(rows)
    frame.to_csv(ARGS.output, index=False)
    cols = [c for c in frame.columns if c[:2] in ("V0", "V1", "V2")]
    print(frame.groupby(["AR", "alpha"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
