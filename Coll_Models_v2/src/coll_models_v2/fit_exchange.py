"""Weighted identification of the energy-partition kernel.

Three kernel forms are available. ``sinkhorn_bridge_v2`` remains the default
and the exact equilibrium model; rejected nonequilibrium nodes can explicitly
select ``conditional_logit_cubic_v3`` without changing accepted nodes.

**The bridge (equilibrium and accepted nodes).** The exchange kernel is an
reversible with respect to Beta(2,2) by construction:

    p(z' | z, eps)  proportional to  6 z'(1-z') h(z) h(z')
                    * exp(lambda3 z z' + lambda1 z' + lambda2 z'^2 + lambda4 eps z')

``h`` is the symmetric Sinkhorn potential of the coupling ``exp(lambda3 z z')``.
With the tilt held at zero the invariant law *is* equipartition -- exactly, to
machine precision, for any memory ``lambda3``.  Elastically there is no
dissipation, so the tilt is held at zero and the elastic limit becomes an
identity of the kernel rather than a number inferred from data.  Only the
dissipative tilt, which is fitted, can move the fixed point.

That matters because the elastic limit is the one place the closure has an
exact answer to be held to.  Fitting it freely, the sentinel put the invariant
mean 1 to 3 percent off equipartition at most nodes and 29 percent off at
(alpha=1, theta=0.2, AR=1.1) -- departures of 4 to 48 sigma at 200k hits.  Held
out on the same shards the bridge costs nothing for that: it *wins* by 0.001 to
0.005 nats at AR 1.1 with one parameter instead of three, and loses by at most
0.003 nats at AR 2 and 3.  Inelastically the two agree to 0.0001 nats and their
invariant means agree to four decimals, so the constraint is inert exactly
where the physics does not demand it.

Note that this makes ``theta*(alpha=1) = 1`` an identity rather than a
validation.  The honest accuracy statement for the elastic limit is the
*unconstrained* fit's departure, which is what the conditional form below
reports and what the sentinel record retains.

**The conditional I-projection (retained for comparison).**  The I-projection of
the memoryless Borgnakke-Larsen draw onto the measured transfer moments:

    p(z' | z, eps)  proportional to  Beta(2,2)(z')
                    * exp(lambda1 z' + lambda2 z'^2 + lambda3 z z' + lambda4 eps z')

Relative to the gated form it previously used, this adds the memory statistic
``z z'`` and the fractional-loss statistic ``eps z'`` to the sufficient
statistics.  Three consequences matter:

* there is no atom at ``z' = z``.  The stored events do not have one -- only 3
  to 13 percent of collisions leave the partition unchanged to 0.01, where the
  Bernoulli form needs 25 to 95 percent -- and forcing one imposes a floor on
  the conditional variance that the data fall below;
* the dual is strictly convex on sample moments, so the "infeasible reset
  moments" branch cannot occur;
* the rotational collision number survives as the derived lag-one slope rather
  than as a parameter the reset law has to be divided by.

``reset_mean`` and ``reset_second_moment`` are retained under their old names
for artifact compatibility, but they now report the first two moments of the
kernel's *invariant* law -- the partition the kernel drives towards, which is
what the reset law was a proxy for.  On synthetic gated-Beta data the two agree
to 0.001 at an exchange probability of 0.4 and to 0.03 at 0.2: the gate is not
a member of this family, and the weaker the exchange the fewer collisions carry
information about the law being approached.

**The bounded-logit conditional form (targeted repair).** At rejected
far-from-equilibrium nodes, a cubic in a standardized, bounded log energy ratio
resolves nonlinear memory without forcing detailed balance on a scalar
marginal process. Its implementation and rationale are documented alongside
``logit_memory_basis`` below.
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import minimize_scalar

from .projections import (
    PROJECTION_TOLERANCE,
    fit_energy_projection,
    _bridge_terms,
    bridge_logpdf,
    bridge_stationary,
    conditional_energy_logpdf,
    conditional_energy_stationary,
    fit_conditional_energy_projection,
)


MODEL_FORM_TOLERANCE_NATS = 0.02
ELASTIC_LOSS_THRESHOLD = 1.0e-9
MEMORY_BOUNDS = (1.0e-6, 900.0)
# Warm-started refits (bootstrap replicates, held-out model form) search a
# multiplicative bracket around the point estimate instead of all nine decades.
# This keeps lambda3 free -- so its bootstrap spread is real rather than
# collapsed to zero -- while cutting the profile search from ~33 evaluations to
# ~10, each starting from a converged tilt.
WARM_BRACKET = (0.4, 2.5)
BRIDGE_BACKTRACKS = 20
BRIDGE_DAMPING_ESCALATIONS = 6


def measure_anchor(z_in: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """Two-moment I-projection of a partition sample.

    The bridge is reversible with respect to this law, so it must be the
    EQUILIBRIUM the kernel relaxes to -- not whatever law the node happens to
    have been generated at. Measured on the elastic, equipartitioned shard it
    is the right thing; measured on an off-equilibrium node it declares that
    node stationary and destroys the restoring force. The elastic data is
    unambiguous about this: at AR 3 one collision moves <z> by +0.2086 from
    theta = 0.2 and by -0.1078 from theta = 2, and by -0.0004 from theta = 1.

    So callers fitting an off-equilibrium node must pass the equilibrium anchor
    in; only the theta = 1 node may measure its own.
    """
    weight = np.asarray(weight, dtype=float)
    weight = weight / np.sum(weight)
    first = float(weight @ z_in)
    second = float(weight @ (z_in * z_in))
    projection = fit_energy_projection(first, second)
    return (float(projection.parameters[0]), float(projection.parameters[1]))


def _bridge_tilt(memory, z_out, z_in, weight, loss, width, quadrature,
                 initial=None, tolerance=1.0e-11, max_iterations=60, anchor=(0.0, 0.0)):
    """Newton on the dissipative tilt at fixed memory.

    At fixed memory the tilt is an ordinary exponential family in
    (z, z**2, loss*z) over the bridge base measure, so the Hessian is again the
    model covariance and Newton converges quadratically.
    """
    if width == 0:
        score = float(weight @ bridge_logpdf(np.array([memory]), z_in, z_out,
                                             loss, quadrature, anchor))
        return np.zeros(0), score, 0.0

    columns = [z_out, z_out * z_out]
    if width > 2:
        columns.append(loss * z_out)
    statistics = np.column_stack(columns[:width])
    target = weight @ statistics
    tilt = np.zeros(width) if initial is None else np.asarray(initial, dtype=float)
    if tilt.shape != (width,):
        tilt = np.zeros(width)

    def dual(value):
        log_norm, _ = _bridge_terms(np.append(memory, value), z_in, loss,
                                    quadrature, order=1, anchor=anchor)
        return float(weight @ log_norm) - float(value @ target)

    def gradient_and_hessian(value):
        _, moments = _bridge_terms(np.append(memory, value), z_in, loss,
                                   quadrature, order=4, anchor=anchor)
        m1, m2, m3, m4 = moments
        scale = [np.ones_like(m1), np.ones_like(m1)]
        if width > 2:
            scale.append(loss)
        scale = np.column_stack(scale[:width])
        model = np.array([weight @ (scale[:, k] * (m2 if k == 1 else m1))
                          for k in range(width)])
        v11, v12, v22 = m2 - m1 * m1, m3 - m1 * m2, m4 - m2 * m2
        curvature = np.empty((width, width))
        for i in range(width):
            for j in range(width):
                block = v22 if (i == 1 and j == 1) else (
                    v12 if (i == 1) != (j == 1) else v11)
                curvature[i, j] = weight @ (scale[:, i] * scale[:, j] * block)
        return model - target, curvature

    # Levenberg-Marquardt damping.  A fixed micro-ridge leaves the Newton step
    # enormous whenever the model covariance is near singular -- which it is
    # when the memory pins z' tightly to z -- and every iteration then pays for
    # ~35 backtracking halvings, each a full pass over the events.  One
    # sentinel node spent 2103 of its 2220 seconds inside a single such solve.
    # Damping the step instead of shrinking it after the fact costs one extra
    # solve of a width-by-width system and removes the pathology.
    current = dual(tilt)
    damping = 1.0e-10
    for _ in range(max_iterations):
        grad, curvature = gradient_and_hessian(tilt)
        if np.max(np.abs(grad)) < tolerance:
            break
        scale = max(1.0, float(np.max(np.abs(np.diag(curvature)))))
        backtracks = 0
        for _ in range(BRIDGE_DAMPING_ESCALATIONS):
            step = np.linalg.solve(
                curvature + damping * scale * np.eye(width), grad)
            scaling = 1.0
            accepted = False
            for backtracks in range(BRIDGE_BACKTRACKS):
                candidate = tilt - scaling * step
                trial = dual(candidate)
                if np.isfinite(trial) and trial <= current \
                        - 1.0e-4 * scaling * float(grad @ step):
                    accepted = True
                    break
                scaling *= 0.5
            if accepted:
                break
            damping = min(damping * 100.0, 1.0e6)
        if not accepted:
            break
        # Trust the region that worked: relax the damping when the full Newton
        # step is taken, tighten it when the line search had to work.
        damping = max(damping / 5.0, 1.0e-12) if backtracks == 0 \
            else min(damping * 10.0, 1.0e6)
        tilt, current = candidate, trial
    grad, _ = gradient_and_hessian(tilt)
    score = float(weight @ bridge_logpdf(np.append(memory, tilt), z_in, z_out,
                                         loss, quadrature, anchor))
    return tilt, score, float(np.max(np.abs(grad)))


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * value) / np.sum(weight))


def _affine_memory(z_in: np.ndarray, z_out: np.ndarray,
                   weight: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((np.ones(len(z_in)), z_in))
    lhs = (design * weight[:, None]).T @ design
    rhs = (design * weight[:, None]).T @ z_out
    intercept, coefficient = np.linalg.solve(lhs, rhs)
    return float(intercept), float(coefficient)


def fit_bridge_kernel(z_in, z_out, weight, loss=None, quadrature: int = 128,
                      elastic: bool | None = None,
                      initial: np.ndarray | None = None,
                      anchor: tuple | None = None) -> dict:
    """Fit the Sinkhorn-bridge exchange kernel by profile likelihood.

        p(z' | z, eps) proportional to 6 z'(1-z') h(z) h(z')
                       * exp(lambda3 z z' + lambda1 z' + lambda2 z'^2 + lambda4 eps z')

    ``h`` is the symmetric Sinkhorn potential, which makes the kernel reversible
    with respect to Beta(2,2): with a zero tilt the invariant law *is*
    equipartition, exactly, for any lambda3. The dissipative tilt then breaks
    detailed balance and moves the fixed point.

    Elastically there is no dissipation, so the tilt is held at zero and the
    elastic limit becomes an identity rather than something inferred from a
    kernel that is nearly the identity.
    """
    weight = weight / np.sum(weight)
    if anchor is None:
        anchor = measure_anchor(z_in, weight)
    mean_loss = 0.0 if loss is None else float(weight @ loss)
    if elastic is None:
        elastic = bool(loss is None or mean_loss <= ELASTIC_LOSS_THRESHOLD)
    spread = 0.0 if loss is None else float(
        np.sqrt(max(weight @ (loss - mean_loss) ** 2, 0.0)))
    loss_deployed = bool((not elastic) and loss is not None and spread > 1.0e-4)
    width = 0 if elastic else (3 if loss_deployed else 2)

    bounds = MEMORY_BOUNDS
    warm = None
    if initial is not None and len(np.atleast_1d(initial)) == width + 1:
        initial = np.asarray(initial, dtype=float)
        centre = float(initial[0])
        warm = initial[1:]
        low = max(MEMORY_BOUNDS[0], centre * WARM_BRACKET[0])
        high = min(MEMORY_BOUNDS[1], centre * WARM_BRACKET[1])
        if low < high:
            bounds = (low, high)

    cache: dict[float, tuple] = {}

    def profile(memory):
        key = round(float(memory), 9)
        if key not in cache:
            previous = cache[min(cache, key=lambda k: abs(k - key))][0] if cache else warm
            cache[key] = _bridge_tilt(key, z_out, z_in, weight, loss, width,
                                      quadrature, initial=previous, anchor=anchor)
        return cache[key]

    search = minimize_scalar(lambda m: -profile(m)[1], bounds=bounds,
                             method="bounded", options={"xatol": 1.0e-4})
    memory = float(search.x)
    tilt, score, residual = profile(memory)
    parameters = np.append(memory, tilt)
    nodes, mass = bridge_stationary(parameters, mean_loss, quadrature, anchor)
    return {
        "kernel_form": "sinkhorn_bridge_v2",
        "anchor_c1": float(anchor[0]),
        "anchor_c2": float(anchor[1]),
        "elastic_block": bool(elastic),
        "lambda3": memory,
        "lambda1": float(tilt[0]) if width >= 1 else 0.0,
        "lambda2": float(tilt[1]) if width >= 2 else 0.0,
        "lambda4": float(tilt[2]) if width >= 3 else 0.0,
        "loss_covariate_deployed": loss_deployed,
        "mean_fractional_loss": mean_loss,
        "stationary_mean": float(mass @ nodes),
        "stationary_second_moment": float(mass @ (nodes * nodes)),
        "projection_residual": residual,
        "log_density": score,
    }


LOGIT_MEMORY_TANH_SCALE = 2.0
LOGIT_CUBIC_KERNEL = "conditional_logit_cubic_v3"
KERNEL_FORMS = ("sinkhorn_bridge_v2", "conditional_iprojection_v2",
                LOGIT_CUBIC_KERNEL)


def logit_memory_basis(z_in: np.ndarray, center: float, scale: float,
                       degree: int = 3) -> np.ndarray:
    """Stable memory coordinates for an energy fraction near zero or one.

    Raw powers of ``z`` become numerically singular as theta tends to zero:
    the fitted linear coefficient reached 2389 on the near-sphere shard and a
    raw cubic still bought 0.038 held-out nats.  The log energy ratio resolves
    multiplicative changes at either boundary. Centering/scaling is measured
    per node, and the hyperbolic tangent maps the unbounded standardized ratio
    to (-1, 1), preventing a single tail event from dominating the exponential
    tilt or the exported table range.
    """
    z = np.clip(np.asarray(z_in, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    standardized = ((np.log(z / (1.0 - z)) - float(center))
                    / max(float(scale), 1.0e-8))
    x = np.tanh(standardized / LOGIT_MEMORY_TANH_SCALE)
    return np.column_stack([x ** power for power in range(1, int(degree) + 1)])


def _basis_stationary(lambda1: float, lambda2: float, coefficients: np.ndarray,
                      center: float, scale: float, offset: float,
                      quadrature: int) -> tuple[np.ndarray, np.ndarray]:
    """Invariant diagnostic for a conditional kernel with logit memory."""
    from .projections import _legendre_nodes
    z, weight = _legendre_nodes(quadrature, 0.0, 1.0)
    basis = logit_memory_basis(z, center, scale, len(coefficients))
    shift = basis @ np.asarray(coefficients, dtype=float) + float(offset)
    log_base = np.log(weight * 6.0 * z * (1.0 - z))
    exponent = (log_base + lambda1 * z + lambda2 * z * z)[None, :] \
        + shift[:, None] * z[None, :]
    exponent -= np.max(exponent, axis=1, keepdims=True)
    transition = np.exp(exponent)
    transition /= np.sum(transition, axis=1, keepdims=True)
    system = np.vstack((transition.T - np.eye(len(z)), np.ones(len(z))))
    rhs = np.zeros(len(z) + 1); rhs[-1] = 1.0
    mass, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    mass = np.maximum(mass, 0.0)
    return z, mass / np.sum(mass)


def _conditional_logit_exchange(z_in, z_out, weight, loss, quadrature,
                                model_form: bool, initial=None,
                                compute_stationary: bool = True) -> dict:
    """Local nonequilibrium kernel with cubic log-energy-ratio memory.

    Detailed balance is an equilibrium property of the full collision state.
    After orientation and impact geometry are marginalized, forcing every
    far-from-equilibrium scalar-z node to be reversible overconstrains the
    local response.  This family is used only at nodes rejected by that bridge;
    the elastic theta=1 anchor remains the reversible Sinkhorn kernel.
    """
    weight = np.asarray(weight, dtype=float)
    weight = weight / np.sum(weight)
    intercept, coefficient = _affine_memory(z_in, z_out, weight)
    p_exch = 1.0 - coefficient

    z_safe = np.clip(np.asarray(z_in, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    logit = np.log(z_safe / (1.0 - z_safe))
    center = float(weight @ logit)
    scale = float(np.sqrt(max(weight @ ((logit - center) ** 2), 1.0e-16)))
    cubic = logit_memory_basis(z_in, center, scale, degree=3)

    mean_loss = 0.0 if loss is None else float(weight @ loss)
    spread = 0.0 if loss is None else float(
        np.sqrt(max(weight @ (loss - mean_loss) ** 2, 0.0)))
    loss_deployed = bool(loss is not None and spread > 1.0e-4)
    columns = [cubic[:, k] for k in range(3)]
    names = ["logit_z", "logit_z2", "logit_z3"]
    if loss_deployed:
        columns.append(np.asarray(loss, dtype=float))
        names.append("loss")
    design = np.column_stack(columns)
    expected_width = 2 + design.shape[1]
    warm = np.asarray(initial, dtype=float) if initial is not None else None
    if warm is not None and warm.shape != (expected_width,):
        warm = None
    projection = fit_conditional_energy_projection(
        z_out, design, weight, tuple(names), quadrature=quadrature, initial=warm)

    base_score = rich_score = float("nan")
    gain = 0.0
    if model_form:
        train = np.arange(len(z_in)) % 5 != 0
        test = ~train
        quartic = logit_memory_basis(z_in, center, scale, degree=4)
        rich_columns = [quartic[:, k] for k in range(4)]
        if loss_deployed:
            rich_columns.append(np.asarray(loss, dtype=float))
        enriched = np.column_stack(rich_columns)
        base = fit_conditional_energy_projection(
            z_out[train], design[train], weight[train], quadrature=quadrature,
            initial=projection.parameters)
        rich_initial = np.insert(projection.parameters, 5, 0.0)
        rich = fit_conditional_energy_projection(
            z_out[train], enriched[train], weight[train], quadrature=quadrature,
            initial=rich_initial)
        base_score = _weighted_mean(conditional_energy_logpdf(
            base.parameters, design[test], z_out[test], quadrature), weight[test])
        rich_score = _weighted_mean(conditional_energy_logpdf(
            rich.parameters, enriched[test], z_out[test], quadrature), weight[test])
        gain = float(rich_score - base_score)

    parameters = projection.parameters
    memory = np.asarray(parameters[2:5], dtype=float)
    loss_coefficient = float(parameters[5]) if loss_deployed else 0.0
    if compute_stationary:
        nodes, mass = _basis_stationary(
            float(parameters[0]), float(parameters[1]), memory, center, scale,
            loss_coefficient * mean_loss, quadrature)
        stationary_mean = float(mass @ nodes)
        stationary_second = float(mass @ (nodes * nodes))
    else:
        # Bootstrap precision is attached to the deployed natural parameters.
        # Re-solving a dense stationary system in every replicate would add no
        # runtime-relevant uncertainty and dominate the nine targeted refits.
        stationary_mean = stationary_second = float("nan")
    return {
        "p_exch": float(p_exch),
        # Retained as a diagnostic only.  The deployed continuous kernel uses
        # the memory coefficients directly; it has no Bernoulli exchange gate.
        "memory_diagnostic_pass": bool(0.0 < p_exch <= 1.0),
        "affine_intercept": float(intercept),
        "affine_slope": float(coefficient - 1.0),
        "lambda1": float(parameters[0]),
        "lambda2": float(parameters[1]),
        "lambda3": float(memory[0]),
        "lambda4": loss_coefficient,
        "lambda5": float(memory[1]),
        "lambda6": float(memory[2]),
        "memory_coefficients": memory.tolist(),
        "memory_coordinate": "bounded_standardized_logit_z",
        "memory_center": center,
        "memory_scale": scale,
        "memory_tanh_scale": LOGIT_MEMORY_TANH_SCALE,
        "loss_covariate_deployed": loss_deployed,
        "mean_fractional_loss": mean_loss,
        "mean_partition_out": _weighted_mean(z_out, weight),
        "stationary_mean": stationary_mean,
        "stationary_second_moment": stationary_second,
        "reset_mean": stationary_mean,
        "reset_second_moment": stationary_second,
        "projection_residual": float(projection.residual),
        "projection_converged": bool(projection.converged),
        "heldout_base_log_density": float(base_score),
        "heldout_enriched_log_density": float(rich_score),
        "nonlinear_improvement": gain,
        "model_form_pass": bool(not model_form or gain < MODEL_FORM_TOLERANCE_NATS),
        "elastic_block": False,
        "anchor_c1": 0.0,
        "anchor_c2": 0.0,
        "kernel_form": LOGIT_CUBIC_KERNEL,
    }


def _bridge_exchange(z_in, z_out, weight, loss, quadrature,
                     model_form: bool, initial=None, anchor=None) -> dict:
    """``fit_exchange_kernel`` contract, served by the Sinkhorn bridge."""
    intercept, coefficient = _affine_memory(z_in, z_out, weight)
    p_exch = 1.0 - coefficient
    if anchor is None:
        anchor = measure_anchor(z_in, weight)
    fit = fit_bridge_kernel(z_in, z_out, weight, loss=loss, quadrature=quadrature,
                            initial=initial, anchor=anchor)
    point = [fit["lambda3"]]
    if not fit["elastic_block"]:
        point += [fit["lambda1"], fit["lambda2"]]
        if fit["loss_covariate_deployed"]:
            point += [fit["lambda4"]]
    point = np.asarray(point, dtype=float)

    # Held-out model form.  The enriched family is the *unconstrained*
    # conditional tilt: the question the gate asks is whether imposing
    # equipartition costs real fit.  On the 200k sentinel the gain is at most
    # 0.003 nats, and it is negative -- the bridge fits better -- at the low
    # aspect ratios where the free fit misidentifies the invariant law.
    base_score = rich_score = float("nan")
    gain = 0.0
    if model_form:
        train = np.arange(len(z_in)) % 5 != 0
        test = ~train
        elastic = fit["elastic_block"]
        held = fit_bridge_kernel(z_in[train], z_out[train], weight[train],
                                 loss=None if loss is None else loss[train],
                                 quadrature=quadrature, elastic=elastic,
                                 initial=point, anchor=anchor)
        block = [held["lambda3"]]
        if not elastic:
            block += [held["lambda1"], held["lambda2"]]
            if held["loss_covariate_deployed"]:
                block += [held["lambda4"]]
        base_score = _weighted_mean(bridge_logpdf(
            np.array(block), z_in[test], z_out[test],
            None if elastic else loss[test], quadrature, anchor), weight[test])
        design = np.column_stack([z_in] + ([] if elastic else [loss]))
        free = fit_conditional_energy_projection(
            z_out[train], design[train], weight[train], quadrature=quadrature)
        rich_score = _weighted_mean(conditional_energy_logpdf(
            free.parameters, design[test], z_out[test], quadrature), weight[test])
        gain = float(rich_score - base_score)

    return {
        "p_exch": float(p_exch),
        "memory_diagnostic_pass": bool(0.0 < p_exch <= 1.0),
        "affine_intercept": float(intercept),
        "affine_slope": float(coefficient - 1.0),
        "lambda1": fit["lambda1"],
        "lambda2": fit["lambda2"],
        "lambda3": fit["lambda3"],
        "lambda4": fit["lambda4"],
        "loss_covariate_deployed": fit["loss_covariate_deployed"],
        "mean_fractional_loss": fit["mean_fractional_loss"],
        "mean_partition_out": _weighted_mean(z_out, weight),
        "stationary_mean": fit["stationary_mean"],
        "stationary_second_moment": fit["stationary_second_moment"],
        "reset_mean": fit["stationary_mean"],
        "reset_second_moment": fit["stationary_second_moment"],
        "projection_residual": fit["projection_residual"],
        "projection_converged": bool(fit["projection_residual"] <= PROJECTION_TOLERANCE),
        "heldout_base_log_density": float(base_score),
        "heldout_enriched_log_density": float(rich_score),
        "nonlinear_improvement": gain,
        "model_form_pass": bool(not model_form or gain < MODEL_FORM_TOLERANCE_NATS),
        "elastic_block": fit["elastic_block"],
        "anchor_c1": fit["anchor_c1"], "anchor_c2": fit["anchor_c2"],
        "kernel_form": "sinkhorn_bridge_v2",
    }


def fit_exchange_kernel(z_in: np.ndarray, z_out: np.ndarray,
                        weight: np.ndarray, loss: np.ndarray | None = None,
                        quadrature: int = 128,
                        loss_spread_threshold: float = 1.0e-4,
                        model_form: bool = True,
                        initial: np.ndarray | None = None,
                        kernel_form: str = "sinkhorn_bridge_v2",
                        anchor: tuple | None = None,
                        compute_stationary: bool = True) -> dict:
    z_in, z_out, weight = map(lambda x: np.asarray(x, dtype=float),
                              (z_in, z_out, weight))
    if not (z_in.shape == z_out.shape == weight.shape) or z_in.ndim != 1:
        raise ValueError("z_in, z_out, and weight must be equal-length vectors")
    if np.any(weight <= 0.0) or np.any((z_in <= 0.0) | (z_in >= 1.0)) \
            or np.any((z_out <= 0.0) | (z_out >= 1.0)):
        raise ValueError("weights must be positive and partitions must lie in (0,1)")
    if kernel_form not in KERNEL_FORMS:
        raise ValueError(f"unknown kernel_form {kernel_form!r}; expected one of {KERNEL_FORMS}")
    if loss is not None:
        loss = np.asarray(loss, dtype=float)
        if loss.shape != z_in.shape:
            raise ValueError("loss must have one entry per event")
    if kernel_form == "sinkhorn_bridge_v2":
        return _bridge_exchange(z_in, z_out, weight, loss, quadrature,
                                model_form, initial, anchor)
    if kernel_form == LOGIT_CUBIC_KERNEL:
        return _conditional_logit_exchange(
            z_in, z_out, weight, loss, quadrature, model_form, initial,
            compute_stationary=compute_stationary)

    # Affine memory diagnostic. This is unchanged and is what the previous
    # p_exch was, but it is now reported rather than inverted for a reset law.
    # It is never clipped, and a value outside (0,1] no longer blocks the fit:
    # the projection is well posed for any conditional law on (0,1), so a
    # diagnostic the kernel does not use must not be able to veto it.  The QA
    # gate carries the flag instead.
    intercept, coefficient = _affine_memory(z_in, z_out, weight)
    p_exch = 1.0 - coefficient
    memory_diagnostic_pass = bool(0.0 < p_exch <= 1.0)

    covariates = [z_in]
    names = ["z_in"]
    loss_deployed = False
    if loss is not None:
        mean_loss = _weighted_mean(loss, weight)
        spread = np.sqrt(max(_weighted_mean((loss - mean_loss) ** 2, weight), 0.0))
        loss_deployed = bool(spread > loss_spread_threshold)
        if loss_deployed:
            covariates.append(loss)
            names.append("loss")
    else:
        mean_loss = 0.0

    design = np.column_stack(covariates)
    projection = fit_conditional_energy_projection(
        z_out, design, weight, tuple(names), quadrature=quadrature, initial=initial)
    if projection.residual > PROJECTION_TOLERANCE:
        raise ValueError(f"energy I-projection residual {projection.residual:.3e}")
    parameters = projection.parameters
    lambda3 = float(parameters[2])
    lambda4 = float(parameters[3]) if loss_deployed else 0.0

    # Invariant law of the fitted kernel at the mean fractional loss.
    nodes, mass = conditional_energy_stationary(
        np.array([parameters[0], parameters[1], lambda3]),
        offset=lambda4 * mean_loss, quadrature=quadrature)
    stationary_mean = float(mass @ nodes)
    stationary_second = float(mass @ (nodes * nodes))

    # Held-out model form. The enriched family adds the quadratic memory
    # statistic z^2 z'; a large gain would mean the mean map is curved in a way
    # the deployed statistics cannot carry.  Bootstrap replicates skip it.
    base_score = rich_score = float("nan")
    gain = 0.0
    if model_form:
        train = np.arange(len(z_in)) % 5 != 0
        test = ~train
        enriched = np.column_stack(covariates + [z_in * z_in])
        base_fit = fit_conditional_energy_projection(
            z_out[train], design[train], weight[train], quadrature=quadrature,
            initial=parameters)
        rich_fit = fit_conditional_energy_projection(
            z_out[train], enriched[train], weight[train], quadrature=quadrature,
            initial=np.append(parameters, 0.0))
        base_score = _weighted_mean(conditional_energy_logpdf(
            base_fit.parameters, design[test], z_out[test], quadrature), weight[test])
        rich_score = _weighted_mean(conditional_energy_logpdf(
            rich_fit.parameters, enriched[test], z_out[test], quadrature), weight[test])
        gain = float(rich_score - base_score)

    return {
        "p_exch": float(p_exch),
        "memory_diagnostic_pass": memory_diagnostic_pass,
        "affine_intercept": float(intercept),
        "affine_slope": float(coefficient - 1.0),
        "lambda1": float(parameters[0]),
        "lambda2": float(parameters[1]),
        "lambda3": lambda3,
        "lambda4": lambda4,
        "loss_covariate_deployed": loss_deployed,
        "mean_fractional_loss": float(mean_loss),
        "mean_partition_out": _weighted_mean(z_out, weight),
        "stationary_mean": stationary_mean,
        "stationary_second_moment": stationary_second,
        # Compatibility aliases; see the module docstring.
        "reset_mean": stationary_mean,
        "reset_second_moment": stationary_second,
        "projection_residual": float(projection.residual),
        "projection_converged": bool(projection.converged),
        "heldout_base_log_density": float(base_score),
        "heldout_enriched_log_density": float(rich_score),
        "nonlinear_improvement": gain,
        "model_form_pass": bool(not model_form or gain < MODEL_FORM_TOLERANCE_NATS),
        "kernel_form": "conditional_iprojection_v2",
    }
