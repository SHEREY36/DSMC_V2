#!/usr/bin/env python3
"""Exercise every compiled Python dependency needed by the HPC pipeline."""

from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np
import scipy
import sklearn
import yaml
from scipy.interpolate import CubicSpline, LinearNDInterpolator, PchipInterpolator
from scipy.optimize import minimize
from scipy.spatial import Delaunay

import coll_models_v2
import dsmc_v2
import dsmc_v2_contracts


def main() -> None:
    locations = {
        "python": sys.executable,
        "scipy": getattr(scipy, "__file__", None),
        "scipy_interpolate": sys.modules["scipy.interpolate"].__file__,
    }
    if not all(locations.values()):
        raise RuntimeError(f"dependency has no concrete installation path: {locations}")

    # Import success alone did not expose the damaged Negishi environment.
    # Exercise the exact interpolation/optimization entry points used later.
    x = np.array([0.0, 0.5, 1.0])
    assert np.isfinite(CubicSpline(x, x * x)(0.25))
    assert np.isfinite(PchipInterpolator(x, x * x)(0.25))
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Delaunay(points)
    assert bool(np.all(np.isfinite(
        LinearNDInterpolator(points, np.arange(3.0))([0.2, 0.2]))))
    assert minimize(lambda value: float(value[0] ** 2), [1.0]).success

    print(json.dumps({
        **locations,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "matplotlib_version": matplotlib.__version__,
        "yaml_version": yaml.__version__,
        "project_imports": [
            dsmc_v2_contracts.__name__, coll_models_v2.__name__, dsmc_v2.__name__],
        "environment_pass": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
