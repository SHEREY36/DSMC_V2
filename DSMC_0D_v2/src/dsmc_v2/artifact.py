"""Strict loaders for the legacy and variational closure artifacts."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay, QhullError

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

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

    def __init__(self, path: str | Path, corrections_enabled: bool = False):
        if corrections_enabled:
            raise ValueError(
                "the invariant (flow-moment) response was retired on 2026-09-30: "
                "the kernel is conditioned on each pair's own state, and the fitted "
                "responses worsened the collisional productions on held-out USF "
                "states; see reports/USF_ENCOUNTER_UNIT_2026-09-29.md")
        data = np.load(path, allow_pickle=False)
        if str(data["schema_version"]) not in ("2.3.0", "2.4.0") \
                or str(data["artifact_type"]) != "bl_variational_closure":
            raise ValueError("not a supported BL variational closure artifact")
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
        # the incoming pair arrives as
        # a = lambda1 + memory(z_in) + lambda4 eps, so the sampler interpolates
        # in a as well as in the uniform draw.
        stored_tables = np.asarray(data["energy_quantiles"], dtype=float)
        stored_a_grid = np.asarray(data["energy_a_grid"], dtype=float)
        if "energy_a_offsets" in data.files:
            offsets = np.asarray(data["energy_a_offsets"], dtype=np.int64)
            if offsets.shape != (len(self.coordinates) + 1,) \
                    or offsets[0] != 0 or offsets[-1] != len(stored_a_grid) \
                    or np.any(np.diff(offsets) < 2) \
                    or stored_a_grid.ndim != 1 \
                    or stored_tables.shape != (len(stored_a_grid), len(self.probability)):
                raise ValueError("packed adaptive energy table layout is invalid")
            self.energy_a_grid = [stored_a_grid[offsets[i]:offsets[i + 1]]
                                  for i in range(len(self.coordinates))]
            self.energy_tables = [stored_tables[offsets[i]:offsets[i + 1]]
                                  for i in range(len(self.coordinates))]
            if "energy_logit_sensitivities" in data.files:
                packed_sensitivity = np.asarray(
                    data["energy_logit_sensitivities"], dtype=float)
                if packed_sensitivity.shape[:1] != stored_tables.shape[:1] \
                        or packed_sensitivity.ndim != 3 \
                        or packed_sensitivity.shape[2] != len(self.probability):
                    raise ValueError("packed energy sensitivity layout is invalid")
                self.energy_logit_sensitivities = [
                    packed_sensitivity[offsets[i]:offsets[i + 1]]
                    for i in range(len(self.coordinates))]
            else:
                self.energy_logit_sensitivities = None
            self.energy_table_layout = "packed_adaptive_v1"
        else:
            # Backward-compatible reader for already deployed rectangular
            # schema-2.3 artifacts.
            self.energy_tables = stored_tables
            self.energy_a_grid = stored_a_grid
            self.energy_table_layout = "rectangular_v1"
            self.energy_logit_sensitivities = (
                np.asarray(data["energy_logit_sensitivities"], dtype=float)
                if "energy_logit_sensitivities" in data.files else None)
        self.energy_sensitivity_parameter_names = tuple(
            data["energy_sensitivity_parameter_names"].astype(str)
            if "energy_sensitivity_parameter_names" in data.files else ())
        self.kernel_form = str(data["kernel_form"])
        self.energy_kernel_forms = (
            np.asarray(data["energy_kernel_forms"]).astype(str)
            if "energy_kernel_forms" in data.files
            else np.full(len(self.coordinates), self.kernel_form))
        self.energy_memory_coefficients = (
            np.asarray(data["energy_memory_coefficients"], dtype=float)
            if "energy_memory_coefficients" in data.files
            else np.column_stack((self.energy_parameters[:, 2],
                                  np.zeros((len(self.coordinates), 2)))))
        self.energy_memory_center = (
            np.asarray(data["energy_memory_center"], dtype=float)
            if "energy_memory_center" in data.files
            else np.zeros(len(self.coordinates)))
        self.energy_memory_scale = (
            np.asarray(data["energy_memory_scale"], dtype=float)
            if "energy_memory_scale" in data.files
            else np.ones(len(self.coordinates)))
        supported_energy_forms = {
            "sinkhorn_bridge_v2", "conditional_iprojection_v2",
            "conditional_logit_cubic_v3",
        }
        if self.energy_kernel_forms.shape != (len(self.coordinates),) \
                or not set(self.energy_kernel_forms).issubset(supported_energy_forms):
            raise ValueError("artifact has invalid per-node energy kernel forms")
        if self.energy_memory_coefficients.shape != (len(self.coordinates), 3):
            raise ValueError("artifact energy memory coefficients must be (node, 3)")
        if self.energy_memory_center.shape != (len(self.coordinates),) \
                or self.energy_memory_scale.shape != (len(self.coordinates),) \
                or np.any(~np.isfinite(self.energy_memory_center)) \
                or np.any(~np.isfinite(self.energy_memory_scale)) \
                or np.any(self.energy_memory_scale <= 0.0):
            raise ValueError("artifact energy-memory normalization is invalid")
        self.energy_interpolation = "node_first_quantile_interpolation_v1"
        declared_interpolation = (str(data["energy_interpolation"])
                                  if "energy_interpolation" in data.files else None)
        if declared_interpolation not in (None, self.energy_interpolation):
            raise ValueError(
                f"unsupported energy interpolation {declared_interpolation!r}")
        # Mean fractional loss of the CTC ensemble each node was fitted on.
        # lambda4 multiplies the loss, so the tilt it produces is only right if
        # the loss handed to it is on the same scale it was fitted against.
        self.energy_mean_loss = np.asarray(data["energy_mean_loss"], dtype=float)
        # Reference law the kernel is reversible with respect to. The sampler
        # tables already bake it in; it is carried for diagnostics and gates.
        self.energy_anchor = np.asarray(data["energy_anchor"], dtype=float)
        # Rotational enhancement of the collision measure, normalised to mean
        # one so installing it changes WHICH pairs collide, not how many.
        if "xi_grid" not in data.files or "xi_enhancement" not in data.files:
            raise ValueError(
                "artifact carries no collision-measure enhancement; a variational "
                "artifact must ship xi_grid and xi_enhancement or the runtime "
                "silently samples the wrong collision ensemble")
        self.xi_grid = np.asarray(data["xi_grid"], dtype=float)
        self.xi_enhancement = np.asarray(data["xi_enhancement"], dtype=float)
        if self.xi_enhancement.ndim != 2 \
                or self.xi_enhancement.shape[0] != len(self.coordinates) \
                or self.xi_enhancement.shape[1] != len(self.xi_grid):
            raise ValueError("xi_enhancement must be (node, len(xi_grid)): the "
                             "curve depends on aspect ratio and on theta")
        if self.energy_table_layout == "rectangular_v1":
            if self.energy_tables.ndim != 3 \
                    or self.energy_tables.shape[0] != len(self.coordinates) \
                    or self.energy_tables.shape[1] != self.energy_a_grid.shape[1]:
                raise ValueError("energy quantile table must be (node, a, u)")
            if not np.all(np.diff(self.energy_a_grid, axis=1) > 0.0):
                raise ValueError("energy a-grid must be strictly increasing per node")
        elif any(not np.all(np.diff(grid) > 0.0) for grid in self.energy_a_grid):
            raise ValueError("energy a-grid must be strictly increasing per node")
        self.energy_axis_clamps = 0
        self.energy_monotonic_repairs = 0
        self.maximum_energy_monotonic_repair = 0.0
        # In a 0-D run alpha and aspect ratio are fixed and theta drifts slowly,
        # so the same interpolation stencil and angular state recur thousands
        # of times. Cache the small state on a rounded key. Energy tables stay
        # node-owned and are never copied into this cache.
        self._state_cache: dict[tuple, dict] = {}
        self._state_cache_hits = 0
        self.angular_tables = np.asarray(data["angular_quantiles"], dtype=float)
        # Cell-moment bounds are carried for diagnostics only.
        self.feature_lower = (np.asarray(data["feature_lower"], dtype=float)
                              if "feature_lower" in data.files else None)
        self.feature_upper = (np.asarray(data["feature_upper"], dtype=float)
                              if "feature_upper" in data.files else None)
        self.joint_deployed = np.asarray(data["joint_deployed"], dtype=bool)
        self.joint_parameters = np.asarray(data["joint_parameters"], dtype=float)
        if np.any(~np.isfinite(self.p_exch)):
            raise ValueError("artifact contains a non-finite affine-memory diagnostic")
        if not np.all(np.diff(self.probability) > 0.0) \
                or self.probability[0] != 0.0 or self.probability[-1] != 1.0:
            raise ValueError("artifact quantile axis must increase from zero to one")
        for axis in range(3):
            if not np.all(np.diff(np.unique(self.coordinates[:, axis])) > 0.0):
                raise ValueError("artifact physical grid axes must be strictly monotone")
        self.corrections_enabled = False
        # Kept at zero for callers that still read these counters.
        self.out_of_domain_queries = 0
        self.total_queries = 0
        self.out_of_domain_by_feature = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
        self.sampling_excursion_by_feature = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
        self._interpolators = {}
        self._coordinate_index = {
            tuple(float(value) for value in row): index
            for index, row in enumerate(self.coordinates)
        }
        self._coordinate_axes = tuple(
            np.unique(self.coordinates[:, axis]) for axis in range(3))
        self._physical_triangulation = None
        if len(self.coordinates) >= 4:
            try:
                self._physical_triangulation = Delaunay(self.coordinates)
            except QhullError:
                # Exact nodes and complete tensor cells still work for a
                # lower-dimensional test/design.  Off-grid queries fail
                # closed in _physical_vertex_weights below.
                self._physical_triangulation = None
        if len(self.coordinates):
            self._interpolators["joint_mask"] = NearestNDInterpolator(
                self.coordinates, self.joint_deployed.astype(float))
        joint_coordinates = self.coordinates[self.joint_deployed]
        if len(joint_coordinates) >= 4:
            try:
                self._interpolators["joint_parameters"] = LinearNDInterpolator(
                    joint_coordinates, self.joint_parameters[self.joint_deployed])
            except QhullError:
                # A grid confined to one alpha (or theta, or AR) plane is flat in
                # 3-D; exact-plane queries use the node stencil and need no
                # triangulation.
                pass

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

    def _physical_vertex_weights(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return one common interpolation stencil for a physical query.

        A complete Cartesian cell is interpolated multilinearly.  This is
        important on the sampled alpha/theta/AR grid: asking at an exact alpha
        or aspect ratio must not mix the adjacent planes merely because a
        Delaunay tessellation chose a diagonal through the cube.  Incomplete
        cells (for example a deliberately truncated low-theta campaign) fall
        back to barycentric interpolation on the measured convex hull.
        """
        exact_index = self._coordinate_index.get(
            tuple(float(value) for value in query))
        if exact_index is not None:
            return np.array([exact_index], dtype=int), np.ones(1, dtype=float)

        choices = []
        for axis in range(3):
            values = self._coordinate_axes[axis]
            matched = np.flatnonzero(np.isclose(values, query[axis], atol=1.0e-12,
                                                 rtol=0.0))
            if len(matched):
                choices.append([(float(values[matched[0]]), 1.0)])
                continue
            upper = int(np.searchsorted(values, query[axis]))
            if upper == 0 or upper == len(values):
                raise ValueError("physical query lies outside the calibrated hull")
            lower = upper - 1
            span = float(values[upper] - values[lower])
            high_weight = float((query[axis] - values[lower]) / span)
            choices.append([(float(values[lower]), 1.0 - high_weight),
                            (float(values[upper]), high_weight)])

        tensor_indices, tensor_weights = [], []
        tensor_complete = True
        for vertex in itertools.product(*choices):
            coordinate = tuple(item[0] for item in vertex)
            index = self._coordinate_index.get(coordinate)
            if index is None:
                tensor_complete = False
                break
            tensor_indices.append(index)
            tensor_weights.append(float(np.prod([item[1] for item in vertex])))
        if tensor_complete:
            return (np.asarray(tensor_indices, dtype=int),
                    np.asarray(tensor_weights, dtype=float))

        tri = self._physical_triangulation
        if tri is None:
            raise ValueError("physical query has no complete interpolation cell")
        simplex = int(tri.find_simplex(query))
        if simplex < 0:
            raise ValueError("physical query lies outside the calibrated hull")
        transform = tri.transform[simplex]
        leading = transform[:3] @ (query - transform[3])
        weights = np.r_[leading, 1.0 - np.sum(leading)]
        if np.any(weights < -1.0e-10):
            raise ValueError("physical interpolation produced invalid barycentric weights")
        weights = np.maximum(weights, 0.0)
        weights /= np.sum(weights)
        return tri.simplices[simplex].astype(int), weights

    @staticmethod
    def _weighted(values: np.ndarray, indices: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
        return np.tensordot(weights, np.asarray(values)[indices], axes=(0, 0))

    def physical_axis_bounds(self, axis: int) -> tuple[float, float]:
        """Return the closed calibrated bounds of one physical coordinate."""
        values = self._coordinate_axes[int(axis)]
        return float(values[0]), float(values[-1])

    def physical_theta_bounds(self, alpha: float,
                              aspect_ratio: float) -> tuple[float, float]:
        """Return theta support at one (possibly interpolated) alpha and AR."""
        supported = []
        for theta in self._coordinate_axes[1]:
            try:
                self._physical_vertex_weights(np.array(
                    [float(alpha), float(theta), float(aspect_ratio)]))
            except ValueError:
                continue
            supported.append(float(theta))
        if not supported:
            raise ValueError(
                "physical alpha/aspect-ratio query has no calibrated theta support")
        return min(supported), max(supported)

    STATE_CACHE_LIMIT = 4096

    def kernel_state(self, alpha: float, theta: float, aspect_ratio: float,
                     features: np.ndarray | None = None,
                     domain_features: np.ndarray | None = None) -> dict:
        """Collision-law state at (alpha, theta, AR).

        On a fitted node the node's own laws are used; elsewhere the stencil
        of Section 14 mixes neighbouring node laws.  The law depends on the
        cell only through (alpha, theta, AR): every pair-level dependence is
        carried by the pair's own variables inside the kernels.  ``features``
        is accepted for call compatibility and ignored.
        """
        self.total_queries += 1
        key = (round(float(alpha), 9), round(float(theta), 3),
               round(float(aspect_ratio), 9))
        hit = self._state_cache.get(key)
        if hit is not None:
            self._state_cache_hits += 1
            return hit
        query = np.array([alpha, theta, aspect_ratio], dtype=float)
        vertex_indices, vertex_weights = self._physical_vertex_weights(query)
        weighted = lambda values: self._weighted(values, vertex_indices, vertex_weights)
        exact = (vertex_indices[:1]
                 if len(vertex_indices) == 1
                 and np.all(np.isclose(self.coordinates[vertex_indices[0]], query,
                                       atol=1.0e-12, rtol=0.0))
                 else np.empty(0, dtype=int))
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
        state = {"p_exch": float(weighted(self.p_exch)),
                 "energy_parameters": weighted(self.energy_parameters).astype(float),
                 "energy_vertex_indices": vertex_indices,
                 "energy_vertex_weights": vertex_weights,
                 "parameter_correction": {}, "energy_correction": 0.0,
                 "fitted_mean_loss": float(weighted(self.energy_mean_loss)),
                 "energy_anchor": weighted(self.energy_anchor).astype(float),
                 "xi_enhancement": weighted(self.xi_enhancement).astype(float),
                 "angular_parameters": weighted(self.angular_parameters).astype(float),
                 "angular_quantiles": weighted(self.angular_tables).astype(float),
                 "out_of_domain": False, "correction_fallback": False,
                 "energy_corrected": False,
                 "joint_deployed": joint, "joint_parameters": joint_parameters}
        if len(self._state_cache) >= self.STATE_CACHE_LIMIT:
            self._state_cache.clear()
        self._state_cache[key] = state
        return state

    def _energy_quantile_row(self, state: dict, z_in: float, loss: float,
                             loss_mean: float = 0.0) -> np.ndarray:
        """Evaluate at each fitted node, then interpolate the distributions.

        Interpolating lambda, the a-grid and a quantile table independently is
        not equivalent because the conditional law is nonlinear in all three.
        That old order manufactured two extra HCS fixed points near theta=1.
        Blending the neighbouring conditional quantiles is a one-dimensional
        Wasserstein interpolation and remains a valid monotone quantile law.
        It is the interpolation actually justified by the exported tables.
        """
        indices = np.asarray(state["energy_vertex_indices"], dtype=int)
        weights = np.asarray(state["energy_vertex_weights"], dtype=float)
        result = np.zeros_like(self.probability, dtype=float)
        clamped = False
        for index, physical_weight in zip(indices, weights):
            lambda1, _, _, lambda4 = self.energy_parameters[index]
            covariate = float(loss)
            fitted_mean = float(self.energy_mean_loss[index])
            if loss_mean > 0.0 and fitted_mean > 0.0:
                covariate *= fitted_mean / float(loss_mean)
            coefficients = self.energy_memory_coefficients[index]
            if self.energy_kernel_forms[index] == "conditional_logit_cubic_v3":
                z = min(max(float(z_in), 1.0e-12), 1.0 - 1.0e-12)
                scale = max(float(self.energy_memory_scale[index]), 1.0e-8)
                standardized = ((np.log(z / (1.0 - z))
                                 - float(self.energy_memory_center[index])) / scale)
                x = float(np.tanh(standardized / 2.0))
                memory = float(coefficients @ np.array([x, x * x, x * x * x]))
            else:
                memory = float(coefficients[0] * float(z_in))
            a = float(lambda1 + memory + lambda4 * covariate)
            grid = self.energy_a_grid[index]
            if a < grid[0] or a > grid[-1]:
                clamped = True
                a = min(max(a, grid[0]), grid[-1])
            upper = int(np.searchsorted(grid, a).clip(1, len(grid) - 1))
            lower = upper - 1
            span = grid[upper] - grid[lower]
            blend = 0.0 if span <= 0.0 else (a - grid[lower]) / span
            table = self.energy_tables[index]
            result += physical_weight * ((1.0 - blend) * table[lower] + blend * table[upper])
        self.energy_axis_clamps += int(clamped)
        return result

    def mean_energy(self, state: dict, z_in: float, loss: float,
                    loss_mean: float = 0.0) -> float:
        """Conditional mean of the exact quantile law deployed at runtime."""
        row = self._energy_quantile_row(state, z_in, loss, loss_mean)
        return float(_trapezoid(row, self.probability))

    def sample_energy(self, state: dict, z_in: float, loss: float,
                      rng: np.random.Generator, loss_mean: float = 0.0) -> float:
        """Draw the outgoing partition from the memory kernel.

        ``z_in`` is the pair's incoming translational share and ``loss`` the
        fractional energy the collision removes.  They enter only through
        ``a = lambda1 + memory(z_in) + lambda4 loss``; the table is indexed by
        that scalar, so the draw is a bilinear interpolation and costs the same
        as the memoryless one it replaces.
        """
        # The loss has two jobs and they must not be confused. How much energy
        # is destroyed uses the raw BL draw. The row helper rescales only the
        # routing covariate onto the CTC loss scale used to fit lambda4.
        row = self._energy_quantile_row(state, z_in, loss, loss_mean)
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
