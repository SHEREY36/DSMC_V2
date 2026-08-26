import unittest

import numpy as np

from coll_models_v2.surfaces import fit_surface


class SurfaceTests(unittest.TestCase):
    def test_surface_interpolates_smooth_function(self):
        x, y = np.meshgrid(np.linspace(0, 1, 7), np.linspace(-1, 1, 6), indexing="ij")
        coordinates = np.column_stack((x.ravel(), y.ravel()))
        values = 1.0 + 0.4 * coordinates[:, 0] - 0.2 * coordinates[:, 1] ** 2
        surface = fit_surface(coordinates, values, ["x", "y"], ridge=1.0e-10)
        predicted = surface.evaluate(coordinates)
        self.assertLess(np.max(np.abs(predicted - values)), 1.0e-5)
        with self.assertRaises(ValueError):
            surface.evaluate([[1.1, 0.0]])


if __name__ == "__main__":
    unittest.main()

