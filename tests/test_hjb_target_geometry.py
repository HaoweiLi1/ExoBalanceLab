"""Check physical box signs and interpolated goal widths independently."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from exobalancelab.hjb_target_geometry import box_value


def test_same_box_signs_and_exact_centerline_crossings_when_underresolved():
    axes = [np.linspace(-0.1, 0.1, 5)] * 3
    widths = np.array([0.025, 0.01, 0.002])
    states = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    original = np.max(np.abs(states) - widths, axis=-1)
    encoded = box_value(axes, np.zeros(3), widths)
    np.testing.assert_array_equal(original <= 0, encoded <= 0)
    assert np.count_nonzero(encoded < 0) == 1
    interpolator = RegularGridInterpolator(axes, encoded)
    for axis in range(3):
        for sign in [-1, 1]:
            face = np.zeros(3)
            face[axis] = sign * widths[axis]
            np.testing.assert_allclose(interpolator(face), 0, atol=1e-16)


def test_physical_box_on_anisotropic_grid_and_shifted_center():
    axes = [np.linspace(-1, 2, 9), np.linspace(-2, 4, 11), np.linspace(-5, 3, 13)]
    center = np.array([0.3, -0.7, 1.1])
    widths = np.array([0.6, 0.8, 1.4])
    states = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    expected = np.all(np.abs(states - center) <= widths, axis=-1)
    np.testing.assert_array_equal(box_value(axes, center, widths) <= 0, expected)
