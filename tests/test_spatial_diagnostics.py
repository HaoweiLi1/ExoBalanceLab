"""Independent checks of the spatial reference and immutable acceptance."""

import json

import numpy as np
import pytest
from scipy.optimize import linprog

from examples.hjb_analytic_reference import exact_bounds
from examples.report_spatial import ROOT, report


def test_double_integrator_bounds_match_independent_control_optimization():
    duration, count = 0.5, 400
    dt = duration / count
    weights = (duration - (np.arange(count) + 0.5) * dt) * dt
    for velocity in [-0.5, -0.23, 0.0, 0.17, 0.505]:
        constraints = np.array([np.full(count, dt), np.full(count, -dt)])
        bounds = np.array([0.01 - velocity, 0.01 + velocity])
        minimum = linprog(weights, A_ub=constraints, b_ub=bounds, bounds=(-1, 1), method="highs")
        maximum = linprog(-weights, A_ub=constraints, b_ub=bounds, bounds=(-1, 1), method="highs")
        assert minimum.success and maximum.success
        lower, upper, feasible = exact_bounds(velocity)
        assert feasible
        np.testing.assert_allclose(
            [lower, upper],
            [-0.005 - velocity * duration + maximum.fun, 0.005 - velocity * duration - minimum.fun],
            atol=dt**2 / 2,
            rtol=0,
        )
    assert not exact_bounds(0.511)[2]


def test_spatial_report_rejects_relaxed_acceptance(tmp_path):
    limits = json.loads((ROOT / "experiments/validation_v1.json").read_text())["acceptance"]
    limits["spatial_area_relative_percent"] = 3.0
    (tmp_path / "manifest.json").write_text(json.dumps({"acceptance": limits}))
    with pytest.raises(ValueError, match="original numerical acceptance"):
        report(tmp_path, tmp_path / "report")
