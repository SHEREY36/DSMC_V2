"""Strict loaders for the legacy and variational closure artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from coll_models_v2.direction_library import DirectionLibrary, conditioning
from coll_models_v2.surfaces import SplineSurface
from dsmc_v2_contracts import FEATURE_NAMES, LEGACY_FEATURE_NAMES


class MicroscopicClosure:
    def __init__(self, routing_path: str | Path, vss_path: str | Path,
                 direction_path: str | Path):
        self.routing = json.loads(Path(routing_path).read_text())
        self.vss = json.loads(Path(vss_path).read_text())
        if self.routing.get("schema_version") != "2.1.0" \
                or self.routing.get("artifact_type") != "routing16_v2":
            raise ValueError("not a routing16_v2 artifact")
        if self.routing.get("feature_order") != list(LEGACY_FEATURE_NAMES):
            raise ValueError("routing feature ordering differs from runtime contract")
        if self.vss.get("inputs") != ["alpha", "aspect_ratio"] \
                or "p_eta" not in self.vss.get("forbidden_inputs", []):
            raise ValueError("VSS artifact violates the direction-only contract")
        self.surfaces = {name: SplineSurface.from_dict(payload)
                         for name, payload in self.routing.get("surfaces", {}).items()}
        self.vss_surfaces = {name: SplineSurface.from_dict(payload)
                             for name, payload in self.vss.get("surfaces", {}).items()}
        self.direction = DirectionLibrary.load(str(direction_path))

    def _routing_quantity(self, name: str, alpha: float, theta: float,
                          aspect_ratio: float) -> float:
        hull = self.routing["design_hull"]
        for value, key in ((alpha, "alpha"), (theta, "theta"),
                           (aspect_ratio, "aspect_ratio")):
            if value < hull[key][0] - 1.0e-12 or value > hull[key][1] + 1.0e-12:
                raise ValueError(f"routing query {key}={value} is outside the design hull")
        coordinate = np.array([[1.0 - alpha * alpha, np.log(theta), np.log(aspect_ratio)]])
        if name in self.surfaces:
            return float(self.surfaces[name].evaluate(coordinate)[0])
        nodes = [node for node in self.routing["nodes"] if node["alpha"] < 1.0]
        points = np.array([[1.0 - n["alpha"]**2, np.log(n["theta"]),
                            np.log(n["aspect_ratio"])] for n in nodes])
        values = np.array([n["quantities"][name]["estimate"] for n in nodes])
        exact = np.flatnonzero(np.all(np.isclose(points, coordinate[0], atol=1.0e-12), axis=1))
        if len(exact):
            return float(values[exact[0]])
        if len(points) < 4:
            raise ValueError(f"no exact {name} node and insufficient grid for interpolation")
        value = float(LinearNDInterpolator(points, values)(coordinate[0]))
        if not np.isfinite(value):
            raise ValueError("routing query lies outside the sampled data hull")
        return value

    def routing_fraction(self, alpha: float, theta: float, aspect_ratio: float,
                         features: np.ndarray) -> float:
        fc = self._routing_quantity("F_C", alpha, theta, aspect_ratio)
        beta = np.array([
            self._routing_quantity(f"beta_ctc_{name}", alpha, theta, aspect_ratio)
            for name in LEGACY_FEATURE_NAMES
        ])
        # f_tr is a modal production ratio, not a probability.  The unchanged
        # v1 energy update permits values outside [0,1] to represent energy
        # transfer between modes while preserving the sampled total loss.
        value = fc * (1.0 + float(np.dot(beta, features)))
        if not np.isfinite(value):
            raise ValueError("non-finite 16-moment routing response")
        return float(value)

    def alpha_eff(self, alpha: float, aspect_ratio: float) -> float:
        coordinate = np.array([[1.0 - alpha * alpha, np.log(aspect_ratio)]])
        if "alpha_eff" in self.vss_surfaces:
            return float(self.vss_surfaces["alpha_eff"].evaluate(coordinate)[0])
        rows = self.vss["rows"]
        points = np.array([[row["alpha"], row["aspect_ratio"]] for row in rows])
        values = np.array([row["alpha_eff"]["estimate"] for row in rows])
        exact = np.flatnonzero(np.all(np.isclose(points, [alpha, aspect_ratio], atol=1.0e-12), axis=1))
        if len(exact):
            return float(values[exact[0]])
        value = float(LinearNDInterpolator(points, values)(alpha, aspect_ratio))
        if not np.isfinite(value):
            raise ValueError("VSS query lies outside the sampled (alpha,AR) hull")
        return value

    def spin_directions(self, alpha: float, theta: float, aspect_ratio: float,
                        c1, c2, w1, w2, u1, u2, outgoing_fractions,
                        mass: float, inertia: float, rng: np.random.Generator) -> np.ndarray:
        query, frame = conditioning(alpha, theta, aspect_ratio, c1, c2, w1, w2,
                                    u1, u2, outgoing_fractions, mass, inertia)
        for _ in range(16):
            donor = self.direction.select(query, rng, neighbours=64).reshape(2, 3)
            lab = donor @ frame
            projected = np.array([lab[0] - np.dot(lab[0], u1) * u1,
                                  lab[1] - np.dot(lab[1], u2) * u2])
            if np.all(np.linalg.norm(projected, axis=1) > 1.0e-10):
                return projected / np.linalg.norm(projected, axis=1)[:, None]
        raise RuntimeError("rotational-direction donors were tangent-degenerate")


class VariationalClosure:
    """Runtime view of ``closure_v2.npz`` with strict no-extrapolation rules."""

    def __init__(self, path: str | Path, corrections_enabled: bool = True):
        data = np.load(path, allow_pickle=False)
        if str(data["schema_version"]) != "2.3.0" \
                or str(data["artifact_type"]) != "bl_variational_closure":
            raise ValueError("not a schema-2.3 BL variational closure artifact")
        if list(data["feature_names"].astype(str)) != list(FEATURE_NAMES):
            raise ValueError("variational feature ordering differs from runtime contract")
        self.coordinates = np.asarray(data["surface_coordinates"], dtype=float)
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3 \
                or len(np.unique(self.coordinates, axis=0)) != len(self.coordinates):
            raise ValueError("artifact surface coordinates are invalid or duplicated")
        self.p_exch = np.asarray(data["p_exch"], dtype=float)
        self.energy_parameters = np.asarray(data["energy_parameters"], dtype=float)
        self.angular_parameters = np.asarray(data["angular_parameters"], dtype=float)
        self.probability = np.asarray(data["quantile_probability"], dtype=float)
        # (node, a, u).  The energy kernel has memory: everything it needs from
        # the incoming pair arrives as a = lambda1 + lambda3 z_in + lambda4 eps,
        # so the sampler interpolates in a as well as in the uniform draw.
        self.energy_tables = np.asarray(data["energy_quantiles"], dtype=float)
        self.energy_a_grid = np.asarray(data["energy_a_grid"], dtype=float)
        self.kernel_form = str(data["kernel_form"])
        if self.energy_tables.ndim != 3 \
                or self.energy_tables.shape[1] != self.energy_a_grid.shape[1]:
            raise ValueError("energy quantile table must be (node, a, u)")
        if not np.all(np.diff(self.energy_a_grid, axis=1) > 0.0):
            raise ValueError("energy a-grid must be strictly increasing per node")
        self.energy_axis_clamps = 0
        self.angular_tables = np.asarray(data["angular_quantiles"], dtype=float)
        self.beta_coordinates = np.asarray(data["beta_coordinates"], dtype=float)
        self.beta = np.asarray(data["beta"], dtype=float)
        self.beta_deployed = np.asarray(data["beta_deployed"], dtype=bool)
        self.feature_lower = np.asarray(data["feature_lower"], dtype=float)
        self.feature_upper = np.asarray(data["feature_upper"], dtype=float)
        self.joint_deployed = np.asarray(data["joint_deployed"], dtype=bool)
        self.joint_parameters = np.asarray(data["joint_parameters"], dtype=float)
        if np.any((self.p_exch <= 0.0) | (self.p_exch > 1.0)):
            raise ValueError("artifact contains an invalid direct exchange probability")
        if not np.all(np.diff(self.probability) > 0.0) \
                or self.probability[0] != 0.0 or self.probability[-1] != 1.0:
            raise ValueError("artifact quantile axis must increase from zero to one")
        for axis in range(3):
            if not np.all(np.diff(np.unique(self.coordinates[:, axis])) > 0.0):
                raise ValueError("artifact physical grid axes must be strictly monotone")
        if corrections_enabled and (self.beta_deployed.size == 0 or not np.any(self.beta_deployed)):
            raise ValueError("natural-parameter corrections requested but no beta is deployed")
        self.corrections_enabled = bool(corrections_enabled)
        self.out_of_domain_queries = 0
        self.total_queries = 0
        self._interpolators = {}
        if len(self.coordinates) >= 4:
            self._interpolators.update(
                p_exch=LinearNDInterpolator(self.coordinates, self.p_exch),
                energy_parameters=LinearNDInterpolator(self.coordinates, self.energy_parameters),
                angular_parameters=LinearNDInterpolator(self.coordinates, self.angular_parameters),
                energy_tables=LinearNDInterpolator(self.coordinates, self.energy_tables),
                angular_tables=LinearNDInterpolator(self.coordinates, self.angular_tables),
            )
        if len(self.beta_coordinates) >= 4:
            self._interpolators["beta"] = LinearNDInterpolator(
                self.beta_coordinates, self.beta)
        if len(self.beta_coordinates):
            self._interpolators["beta_mask"] = NearestNDInterpolator(
                self.beta_coordinates, self.beta_deployed.astype(float))
        if len(self.coordinates):
            self._interpolators["joint_mask"] = NearestNDInterpolator(
                self.coordinates, self.joint_deployed.astype(float))
        joint_coordinates = self.coordinates[self.joint_deployed]
        if len(joint_coordinates) >= 4:
            self._interpolators["joint_parameters"] = LinearNDInterpolator(
                joint_coordinates, self.joint_parameters[self.joint_deployed])

    @staticmethod
    def _exact(points: np.ndarray, query: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.all(np.isclose(points, query, atol=1.0e-12, rtol=0.0), axis=1))

    def _interpolate(self, points: np.ndarray, values: np.ndarray, query: np.ndarray,
                     label: str, interpolator=None):
        exact = self._exact(points, query)
        if len(exact):
            return np.asarray(values[exact[0]])
        if len(points) < 4:
            raise ValueError(f"no exact {label} node and insufficient points to interpolate")
        if interpolator is None:
            interpolator = LinearNDInterpolator(points, values)
        result = np.asarray(interpolator(query[None, :]))[0]
        if np.any(~np.isfinite(result)):
            raise ValueError(f"{label} query lies outside the calibrated physical hull")
        return result

    def kernel_state(self, alpha: float, theta: float, aspect_ratio: float,
                     features: np.ndarray) -> dict:
        features = np.asarray(features, dtype=float)
        if features.shape != (len(FEATURE_NAMES),):
            raise ValueError("variational closure requires fourteen cell features")
        self.total_queries += 1
        ood = bool(np.any(features < self.feature_lower) or np.any(features > self.feature_upper))
        self.out_of_domain_queries += int(ood)
        query = np.array([alpha, theta, aspect_ratio], dtype=float)
        p_exch = float(self._interpolate(self.coordinates, self.p_exch, query, "p_exch",
                                         self._interpolators.get("p_exch")))
        eparams = self._interpolate(self.coordinates, self.energy_parameters, query,
                                    "energy parameters",
                                    self._interpolators.get("energy_parameters")).astype(float)
        aparams = self._interpolate(self.coordinates, self.angular_parameters, query,
                                    "angular parameters",
                                    self._interpolators.get("angular_parameters")).astype(float)
        shape = self.energy_tables.shape[1:]
        etable = self._interpolate(
            self.coordinates, self.energy_tables.reshape(len(self.energy_tables), -1),
            query, "energy quantiles",
            self._interpolators.get("energy_tables")).astype(float).reshape(shape)
        agrid = self._interpolate(self.coordinates, self.energy_a_grid, query,
                                  "energy a-grid",
                                  self._interpolators.get("energy_a_grid")).astype(float)
        atable = self._interpolate(self.coordinates, self.angular_tables, query,
                                   "angular quantiles",
                                   self._interpolators.get("angular_tables")).astype(float)
        beta = np.zeros(len(FEATURE_NAMES))
        if self.corrections_enabled:
            beta = self._interpolate(self.beta_coordinates, self.beta,
                                     query, "lambda1 coefficients",
                                     self._interpolators.get("beta")).astype(float)
            exact_beta = self._exact(self.beta_coordinates, query)
            if len(exact_beta):
                deployed = self.beta_deployed[exact_beta[0]]
            else:
                deployed = np.asarray(
                    self._interpolators["beta_mask"](query[None, :]))[0] >= 0.5
            beta *= deployed
            correction = float(beta @ features)
            eparams[0] += correction
        else:
            correction = 0.0
        exact = self._exact(self.coordinates, query)
        joint = bool(len(exact) and self.joint_deployed[exact[0]])
        joint_parameters = self.joint_parameters[exact[0]].copy() if joint else None
        if not len(exact) and "joint_parameters" in self._interpolators:
            masked = bool(np.asarray(
                self._interpolators["joint_mask"](query[None, :]))[0] >= 0.5)
            if masked:
                candidate = np.asarray(
                    self._interpolators["joint_parameters"](query[None, :]))[0]
                if np.all(np.isfinite(candidate)):
                    joint, joint_parameters = True, candidate.astype(float)
        return {"p_exch": p_exch, "energy_parameters": eparams,
                "energy_a_grid": agrid,
                "angular_parameters": aparams, "energy_quantiles": etable,
                "angular_quantiles": atable, "beta": beta, "out_of_domain": ood,
                "energy_corrected": correction != 0.0,
                "joint_deployed": joint, "joint_parameters": joint_parameters}

    def sample_energy(self, state: dict, z_in: float, loss: float,
                      rng: np.random.Generator) -> float:
        """Draw the outgoing partition from the memory kernel.

        ``z_in`` is the pair's incoming translational share and ``loss`` the
        fractional energy the collision removes.  They enter only through
        ``a = lambda1 + lambda3 z_in + lambda4 loss``; the table is indexed by
        that scalar, so the draw is a bilinear interpolation and costs the same
        as the memoryless one it replaces.
        """
        lambda1, _, lambda3, lambda4 = state["energy_parameters"]
        grid = state["energy_a_grid"]
        a = float(lambda1 + lambda3 * float(z_in) + lambda4 * float(loss))
        if a < grid[0] or a > grid[-1]:
            # The natural-parameter correction shifts lambda1 after the table
            # was tabulated, so a can leave the exported span. Clamping is the
            # conservative choice; the count is reported so it cannot hide.
            self.energy_axis_clamps += 1
            a = min(max(a, grid[0]), grid[-1])
        upper = int(np.searchsorted(grid, a).clip(1, len(grid) - 1))
        lower = upper - 1
        span = grid[upper] - grid[lower]
        blend = 0.0 if span <= 0.0 else (a - grid[lower]) / span
        table = state["energy_quantiles"]
        row = (1.0 - blend) * table[lower] + blend * table[upper]
        return float(np.interp(rng.random(), self.probability, row))

    def sample_direction(self, ghat_pre: np.ndarray, state: dict, z: float,
                         rng: np.random.Generator) -> np.ndarray:
        ghat = np.asarray(ghat_pre, dtype=float)
        ghat /= max(np.linalg.norm(ghat), 1.0e-30)
        table = state["angular_quantiles"]
        if state["joint_deployed"]:
            parameter = state["joint_parameters"]
            linear, quadratic = parameter[0] + parameter[2] * z, parameter[1]
            candidates = [-1.0, 1.0]
            if quadratic < 0.0:
                vertex = -linear / (3.0 * quadratic)
                if -1.0 < vertex < 1.0:
                    candidates.append(vertex)
            maximum = max(linear * c + quadratic * 0.5 * (3.0 * c * c - 1.0)
                          for c in candidates)
            for _ in range(10000):
                cosine = 2.0 * rng.random() - 1.0
                exponent = linear * cosine + quadratic * 0.5 * (3.0 * cosine**2 - 1.0)
                if rng.random() <= np.exp(exponent - maximum):
                    break
            else:
                raise RuntimeError("coupled angular rejection sampler failed")
        else:
            cosine = float(np.interp(rng.random(), self.probability, table))
        trial = np.array([1.0, 0.0, 0.0]) if abs(ghat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = trial - np.dot(trial, ghat) * ghat
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ghat, e1)
        phi = 2.0 * np.pi * rng.random()
        result = cosine * ghat + np.sqrt(max(0.0, 1.0 - cosine * cosine)) * (
            np.cos(phi) * e1 + np.sin(phi) * e2)
        return result / np.linalg.norm(result)

    @staticmethod
    def tangent_spin_directions(axes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        directions = []
        for axis in np.asarray(axes):
            for _ in range(16):
                candidate = rng.normal(size=3)
                candidate -= np.dot(candidate, axis) * axis
                norm = np.linalg.norm(candidate)
                if norm > 1.0e-12:
                    directions.append(candidate / norm)
                    break
            else:
                raise RuntimeError("failed to sample a tangent spin direction")
        return np.asarray(directions)

    @property
    def out_of_domain_fraction(self) -> float:
        return self.out_of_domain_queries / max(self.total_queries, 1)
