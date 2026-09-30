"""Conditional angular I-projection with incoming-partition memory (fitting side).

    p(c | z, z') = exp[a c + b P2(c)] / Z(a, b),   c = ghat.ghat'
    a = eta1 + xi z' + rho1 z,     b = eta2 + zeta z' + rho2 z

Weighted maximum likelihood over CTC encounters == the I-projection matching
E[T c], E[T P2] for T in {1, z', z}.  Newton on the concave log-likelihood with
Gauss-Legendre moments of the per-event normalizer.
"""
import numpy as np

XG, WG = np.polynomial.legendre.leggauss(96)
P2G = 0.5 * (3 * XG * XG - 1)


def moments(a, b):
    """E[c], E[P2], Var/Cov of (c, P2) under exp(a c + b P2) on [-1,1]."""
    e = a[:, None] * XG[None, :] + b[:, None] * P2G[None, :]
    e -= e.max(1, keepdims=True)
    w = np.exp(e) * WG[None, :]
    w /= w.sum(1, keepdims=True)
    mc, mp = w @ XG, w @ P2G
    vcc = w @ (XG * XG) - mc * mc
    vpp = w @ (P2G * P2G) - mp * mp
    vcp = w @ (XG * P2G) - mc * mp
    return mc, mp, vcc, vpp, vcp


def log_norm(a, b):
    e = a[:, None] * XG[None, :] + b[:, None] * P2G[None, :]
    mx = e.max(1)
    return mx + np.log(np.exp(e - mx[:, None]) @ WG)


def design(z_in, z_out, form):
    ones = np.ones_like(z_in)
    if form == "joint":        # current production form (z' in the linear term only)
        return np.column_stack([ones, z_out]), np.column_stack([ones])
    if form == "memory":       # incoming-partition memory in both moments
        # runtime order: (eta1, xi, rho1, eta2, zeta, rho2)
        return (np.column_stack([ones, z_out, z_in]),
                np.column_stack([ones, z_out, z_in]))
    raise ValueError(form)


def fit(cosine, z_in, z_out, weight=None, form="memory", iters=80, prior_sd=5.0):
    """MAP I-projection: weighted log-likelihood plus a N(0, prior_sd^2) prior
    on the memory coefficients (everything except the two intercepts).

    Near the sphere z' ~ z, so the z and z' directions are collinear and the
    unpenalised likelihood is flat along their difference.  The prior is
    inert wherever the data identify the coefficients and fixes that
    direction elsewhere.  Its weight is 1/N_eff per event, a true prior.
    """
    c = np.asarray(cosine, float); p2 = 0.5 * (3 * c * c - 1)
    w = np.ones_like(c) if weight is None else np.asarray(weight, float)
    n_eff = w.sum() ** 2 / np.sum(w * w)
    w = w / w.sum()
    A, B = design(z_in, z_out, form)
    na, nb = A.shape[1], B.shape[1]
    penalty = np.ones(na + nb) / (n_eff * prior_sd ** 2)
    penalty[0] = penalty[na] = 0.0          # intercepts eta1, eta2 are free
    theta = np.zeros(na + nb)

    def objective(t):
        return loglik(t, c, p2, A, B, w) - 0.5 * np.sum(penalty * t * t)

    grad = np.full(na + nb, np.inf)
    for _ in range(iters):
        a, b = A @ theta[:na], B @ theta[na:]
        mc, mp, vcc, vpp, vcp = moments(a, b)
        grad = np.r_[A.T @ (w * (c - mc)), B.T @ (w * (p2 - mp))] - penalty * theta
        if np.max(np.abs(grad)) < 1e-11:
            break
        H = np.block([[A.T @ (A * (w * vcc)[:, None]), A.T @ (B * (w * vcp)[:, None])],
                      [B.T @ (A * (w * vcp)[:, None]), B.T @ (B * (w * vpp)[:, None])]])
        H += np.diag(penalty)
        step = np.linalg.solve(H + 1e-14 * np.eye(len(theta)), grad)
        f0 = objective(theta)
        t = 1.0
        while t > 1e-8:
            f1 = objective(theta + t * step)
            if np.isfinite(f1) and f1 >= f0 - 1e-15:
                break
            t *= 0.5
        theta = theta + t * step
    return {"form": form, "theta": theta, "na": na, "grad": float(np.max(np.abs(grad))),
            "loglik": loglik(theta, c, p2, A, B, w), "n_eff": float(n_eff)}


def loglik(theta, c, p2, A, B, w):
    na = A.shape[1]
    a, b = A @ theta[:na], B @ theta[na:]
    return float(np.sum(w * (a * c + b * p2 - log_norm(a, b))))


def sample(model, z_in, z_out, rng):
    """Inverse-CDF draw of c on a fine grid (vectorised, exact to grid error)."""
    A, B = design(np.asarray(z_in, float), np.asarray(z_out, float), model["form"])
    na = model["na"]; th = model["theta"]
    a, b = A @ th[:na], B @ th[na:]
    grid = np.linspace(-1, 1, 801)
    e = a[:, None] * grid[None, :] + b[:, None] * (0.5 * (3 * grid**2 - 1))[None, :]
    e -= e.max(1, keepdims=True)
    pdf = np.exp(e)
    cdf = np.cumsum(0.5 * (pdf[:, 1:] + pdf[:, :-1]), 1)
    cdf = np.hstack([np.zeros((len(a), 1)), cdf]); cdf /= cdf[:, -1:]
    u = rng.random(len(a))
    idx = np.array([np.searchsorted(cdf[i], u[i]) for i in range(len(a))]).clip(1, 800)
    lo = cdf[np.arange(len(a)), idx - 1]; hi = cdf[np.arange(len(a)), idx]
    f = (u - lo) / np.maximum(hi - lo, 1e-300)
    return grid[idx - 1] + f * (grid[idx] - grid[idx - 1])


def mean_p2(model, z_in, z_out):
    A, B = design(np.asarray(z_in, float), np.asarray(z_out, float), model["form"])
    na = model["na"]; th = model["theta"]
    return moments(A @ th[:na], B @ th[na:])[1]
