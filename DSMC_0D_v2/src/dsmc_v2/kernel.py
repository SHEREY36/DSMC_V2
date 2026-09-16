"""The v1 BL/GMM collision kernel with only routing and angle extension points."""

from __future__ import annotations

import time as wallclock

import numpy as np

from .angular import sample_direction


def prepare_theta(theta: float) -> float:
    return float(np.clip(round(theta * 10.0) / 10.0, 0.1, 1.2))


def rotational_collision_number(theta: float, alpha: float) -> float:
    return 1.67 if alpha >= 1.0 else 0.39 * theta**2 + 0.09 * theta + 1.67


def rank0_ftr(c_alpha: float, theta: float) -> float:
    return float(c_alpha) * 3.0 * theta / (3.0 * theta + 2.0)


def chi_hs(mu: float, alpha: float) -> float:
    denominator = np.sqrt(max(1.0 - (1.0 - alpha * alpha) * mu * mu, 1.0e-30))
    cosine = (1.0 - (1.0 + alpha) * mu * mu) / denominator
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _perpendicular(ghat: np.ndarray) -> np.ndarray:
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, ghat)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    result = trial - np.dot(trial, ghat) * ghat
    return result / np.linalg.norm(result)


def legacy_scatter(vrel: np.ndarray, normal: np.ndarray, chi: float,
                   magnitude: float, epsilon: float) -> np.ndarray:
    ghat = vrel / max(np.linalg.norm(vrel), 1.0e-30)
    mu = float(np.clip(np.dot(normal, ghat), 0.0, 1.0))
    tangent = normal - mu * ghat
    tangent = _perpendicular(ghat) if np.linalg.norm(tangent) <= 1.0e-12 \
        else tangent / np.linalg.norm(tangent)
    second = np.cross(ghat, tangent)
    post = (np.cos(chi) * ghat - np.sin(chi)
            * (np.cos(epsilon) * tangent - np.sin(epsilon) * second))
    return magnitude * post / np.linalg.norm(post)


class SpherocylinderKernel:
    def __init__(self, params, models, alpha: float, beta_a: float, beta_b: float,
                 c_alpha: float, closure, routing_mode: str, angular_mode: str,
                 direction_rng: np.random.Generator, vss_rng: np.random.Generator,
                 equilibration_time: float = 0.0,
                 isotropic_epsilon: bool = True):
        self.params, self.models, self.alpha = params, models, float(alpha)
        self.beta_a, self.beta_b, self.c_alpha = beta_a, beta_b, c_alpha
        self.closure, self.routing_mode, self.angular_mode = closure, routing_mode, angular_mode
        self.direction_rng, self.vss_rng = direction_rng, vss_rng
        self.equilibration_time = float(equilibration_time)
        self.isotropic_epsilon = bool(isotropic_epsilon)
        self.loss = models.loss_parameters(alpha, params.aspect_ratio) if alpha < 1.0 else {
            "gamma_max": 0.0, "one_hit_probability": 1.0}
        # Mean of the loss actually drawn here, used only to put the kernel's
        # loss covariate on the scale it was fitted against. The draw itself is
        # untouched, so the cooling rate and the collision clock do not move.
        self.area_mean, self.area_supremum = self._area_constants()
        self.xi_grid = self.xi_enhancement = None
        self.acceptance_ceiling = self.area_supremum
        # candidates must be inflated by this so the rate is unchanged
        self.candidate_inflation = self.area_supremum / max(self.area_mean, 1e-30)
        self.mean_loss_fraction = float(self.loss.get(
            "mean_loss_fraction",
            beta_a / (beta_a + beta_b) * self.loss["gamma_max"]
            * self.loss["one_hit_probability"]))
        self.cell_routing = None
        self.cell_variational = None
        self.negative_energy_repairs = 0
        self.closure_seconds = 0.0

    def set_cell_routing(self, value: float | None) -> None:
        self.cell_routing = value

    def set_cell_variational(self, value: dict | None) -> None:
        self.cell_variational = value

    def _area_constants(self):
        """Isotropic mean and supremum of the cross section, computed once.

        The mean is what the frozen sigma_c stands in for, so dividing by it and
        inflating the candidate count by max/mean leaves the collision rate
        exactly where it was while making the *selection* orientation-correct.
        """
        d = self.params.diameter
        length = (float(self.params.aspect_ratio) - 1.0) * d
        rng = np.random.default_rng(0x5EED)
        u1 = rng.normal(size=(200000, 3)); u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
        u2 = rng.normal(size=(200000, 3)); u2 /= np.linalg.norm(u2, axis=1, keepdims=True)
        g = rng.normal(size=(200000, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        s1 = np.linalg.norm(np.cross(u1, g), axis=1)
        s2 = np.linalg.norm(np.cross(u2, g), axis=1)
        triple = np.abs(np.einsum("ni,ni->n", g, np.cross(u1, u2)))
        area = (np.pi * d * d + 2.0 * d * length * (s1 + s2)
                + length * length * triple)
        supremum = np.pi * d * d + 4.0 * d * length + length * length
        return float(area.mean()), float(supremum)

    def accept_orientation(self, u1: np.ndarray, u2: np.ndarray, ghat: np.ndarray,
                           w1: np.ndarray, w2: np.ndarray, speed: float, rng) -> bool:
        """Second-stage acceptance reproducing the CTC collision measure.

        The static projected area alone is inert: measured against the exact
        encounter propensity it leaves <z> unchanged to four decimals, because
        A_perp is essentially uncorrelated with the energy partition. What
        actually biases which pairs collide is ROTATION -- a fast-spinning rod
        sweeps more volume while closing, so it collides more often, and it
        also carries more rotational energy. That enters through the
        dimensionless rotation number

            Xi = (|w1| + |w2|)(L + D) / |g|

        and A_perp * g(Xi) reproduces the exact propensity's selection to four
        decimals at every node tested.
        """
        curve = None if self.cell_variational is None else \
            self.cell_variational.get("xi_enhancement")
        if self.xi_grid is None or curve is None:
            # Fail closed. Silently accepting every pair reverts the runtime to
            # the orientation-isotropic proposal ensemble, which is precisely
            # the bug the enhancement exists to remove -- and it would do so
            # with no error and a plausible-looking answer.
            raise RuntimeError(
                "variational routing requires the collision-measure enhancement; "
                "the artifact carries no xi_grid/xi_enhancement table")
        if self.area_supremum <= 0.0:
            return True
        reach = float(self.params.aspect_ratio) * self.params.diameter
        xi = ((float(np.linalg.norm(w1)) + float(np.linalg.norm(w2))) * reach
              / max(speed, 1.0e-30))
        # The curve is interpolated per cell: it depends on aspect ratio, and
        # on theta through its own rate normalisation, so a single global table
        # is wrong away from the node it was fitted at.
        enhancement = float(np.interp(xi, self.xi_grid, curve))
        weight = self.projected_excluded_area(u1, u2, ghat) * enhancement
        # Divide by the SAME ceiling the candidate count was inflated with.
        # Using this cell's own maximum here while inflating by the global one
        # leaves a stray rate factor g_max_global / g_max_cell, which unfreezes
        # the collision clock in exactly the cells whose curve is flattest.
        return bool(rng.random() < weight / self.acceptance_ceiling)

    def set_enhancement(self, grid, ceiling: float) -> None:
        """Install the Xi axis and size the candidate inflation.

        The curve itself is per cell and read from cell_variational; only the
        axis and a bound on g are global. Inflating candidates by
        supremum/mean keeps the accepted rate where the frozen clock put it.
        """
        if grid is None:
            self.xi_grid = None
            self.candidate_inflation = 1.0
            self.acceptance_ceiling = self.area_supremum
            return
        self.xi_grid = np.asarray(grid, dtype=float)
        self.acceptance_ceiling = self.area_supremum * float(ceiling)
        self.candidate_inflation = self.acceptance_ceiling / max(
            self.area_mean, 1.0e-30)

    def projected_excluded_area(self, u1: np.ndarray, u2: np.ndarray,
                                ghat: np.ndarray) -> float:
        """Orientation-dependent collision cross section of a spherocylinder pair.

        The NTC clock accepts on sigma_c |g| with sigma_c constant, which selects
        pairs isotropically in orientation. Real collisions do not: a rod broadside
        to the approach presents far more area than one end-on. Fitting the closure
        on the physical collision ensemble and then feeding it an isotropically
        selected pair breaks the elastic invariant by 4 per cent at AR 3, so the
        acceptance has to carry the same weighting the fit was made under.
        """
        d = self.params.diameter
        length = (float(self.params.aspect_ratio) - 1.0) * d
        s1 = float(np.linalg.norm(np.cross(u1, ghat)))
        s2 = float(np.linalg.norm(np.cross(u2, ghat)))
        triple = abs(float(np.dot(ghat, np.cross(u1, u2))))
        return (np.pi * d * d + 2.0 * d * length * (s1 + s2)
                + length * length * triple)

    def collide(self, state, p1: int, p2: int, normal: np.ndarray,
                v1: np.ndarray, v2: np.ndarray, vrel: np.ndarray,
                relative_speed: float, time: float, theta: float) -> int:
        vcom = 0.5 * (v1 + v2)
        v1com, v2com = v1 - vcom, v2 - vcom
        etr_i = 0.5 * self.params.mass * (np.dot(v1com, v1com) + np.dot(v2com, v2com))
        erot_i = state.rotational_energy[p1] + state.rotational_energy[p2]
        total_i = etr_i + erot_i
        if total_i <= 0.0:
            return 0
        eps_tr_i = etr_i / total_i
        eps_r1_i = state.rotational_energy[p1] / erot_i if erot_i > 0.0 else 0.5
        in_equilibration = time < self.equilibration_time
        if self.routing_mode == "variational_v2":
            closure_started = wallclock.perf_counter()
            if self.cell_variational is None:
                raise RuntimeError("variational closure was not evaluated for the current cell")
            # No Bernoulli gate: the fitted kernel is a conditional law for
            # z' given (z, eps), and the memory that the gate used to stand in
            # for is carried by lambda3 inside it. The loss is drawn first
            # because it enters the kernel through lambda4.
            gamma = 0.0 if in_equilibration or self.alpha >= 1.0 else (
                np.random.beta(self.beta_a, self.beta_b) * self.loss["gamma_max"]
                * self.loss["one_hit_probability"])
            eps_tr_f = self.closure.sample_energy(
                self.cell_variational, eps_tr_i, gamma, self.vss_rng,
                loss_mean=self.mean_loss_fraction)
            eps_r1_f = self.vss_rng.random()
            available = total_i * (1.0 - gamma)
            etr_f, erot_f = eps_tr_f * available, (1.0 - eps_tr_f) * available
            if not (etr_f > 0.0 and erot_f > 0.0):
                self.negative_energy_repairs += 1
                raise RuntimeError("variational kernel produced a non-positive modal energy")
            state.rotational_energy[p1] = eps_r1_f * erot_f
            state.rotational_energy[p2] = (1.0 - eps_r1_f) * erot_f
            ghat = vrel / max(relative_speed, 1.0e-30)
            magnitude = 2.0 * np.sqrt(etr_f / self.params.mass)
            direction = self.closure.sample_direction(
                ghat, self.cell_variational, eps_tr_f, self.vss_rng)
            gpost = magnitude * direction
            state.velocity[p1] = vcom + 0.5 * gpost
            state.velocity[p2] = vcom - 0.5 * gpost
            directions = self.closure.tangent_spin_directions(
                np.array([state.axis[p1], state.axis[p2]]), self.direction_rng)
            state.set_spin_directions((p1, p2), directions)
            self.closure_seconds += wallclock.perf_counter() - closure_started
            return 2

        pr = min(1.0 / rotational_collision_number(theta, self.alpha), 0.5)
        draw = np.random.random()
        relax1, relax2 = draw < pr, draw >= pr and draw < 2.0 * pr
        if relax1 or relax2:
            if self.alpha >= 1.0:
                eps_tr_f, sampled = np.random.beta(2.0, 2.0), np.random.random()
            else:
                sampled_pair = self.models.cond_gmm.sample_conditionals(
                    prepare_theta(theta), eps_tr_i, eps_r1_i, 1)[0]
                eps_tr_f, sampled = sampled_pair
            eps_r1_f = sampled if relax1 else 1.0 - sampled
        else:
            eps_tr_f, eps_r1_f = eps_tr_i, eps_r1_i
        gamma = 0.0 if in_equilibration or self.alpha >= 1.0 else (
            np.random.beta(self.beta_a, self.beta_b) * self.loss["gamma_max"]
            * self.loss["one_hit_probability"])
        delta = gamma * total_i
        if self.routing_mode == "legacy_rank0":
            ftr = rank0_ftr(self.c_alpha, theta)
        else:
            if self.cell_routing is None:
                raise RuntimeError("moment16 routing was not evaluated for the current cell")
            ftr = self.cell_routing
        etr_f = eps_tr_f * total_i - ftr * delta
        erot_f = (1.0 - eps_tr_f) * total_i - (1.0 - ftr) * delta
        # Preserve the v1 reservoir handling exactly.
        if etr_f < 0.0:
            erot_f += etr_f
            etr_f = 1.0e-30
        if erot_f < 0.0:
            etr_f += erot_f
            erot_f = 1.0e-30
        state.rotational_energy[p1] = eps_r1_f * erot_f
        state.rotational_energy[p2] = (1.0 - eps_r1_f) * erot_f

        ghat = vrel / max(relative_speed, 1.0e-30)
        magnitude = 2.0 * max(np.sqrt(etr_f / self.params.mass), 1.0e-14)
        if self.angular_mode == "legacy" or in_equilibration:
            mu = abs(float(np.dot(normal, ghat)))
            chi = chi_hs(mu, 1.0 if in_equilibration else self.alpha)
            epsilon = np.random.uniform(0.0, 2.0 * np.pi) if self.isotropic_epsilon else 0.0
            gpost = legacy_scatter(vrel, normal, chi, magnitude, epsilon)
        else:
            direction = sample_direction(ghat,
                self.closure.alpha_eff(self.alpha, self.params.aspect_ratio), self.vss_rng)
            gpost = magnitude * direction
        state.velocity[p1] = vcom + 0.5 * gpost
        state.velocity[p2] = vcom - 0.5 * gpost

        if self.closure is not None:
            directions = self.closure.spin_directions(
                self.alpha, theta, self.params.aspect_ratio, v1, v2,
                state.omega[p1], state.omega[p2], state.axis[p1], state.axis[p2],
                np.array([eps_tr_f, eps_r1_f]), self.params.mass,
                self.params.inertia, self.direction_rng)
            state.set_spin_directions((p1, p2), directions)
        else:
            state.preserve_spin_directions((p1, p2))
        return 2
