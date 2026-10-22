"""Opt-in JAX checks against MATLAB stencils and exact linear advection."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from exobalancelab.paper import PROFILES
from exobalancelab.reachability import HJBPolicy, policy_contract

pytestmark = pytest.mark.skipif(
    os.environ.get("EXOBALANCE_JAX_TEST") != "1", reason="Separate JAX environment is opt-in"
)


def test_eno3_matches_independent_matlab_stencils_including_ties():
    import jax.numpy as jnp
    from hj_reachability.boundary_conditions import extrapolate_away_from_zero

    from exobalancelab.hjb_solver import _eno3

    fixture = json.loads((Path(__file__).parent / "data/eno3_matlab.json").read_text())
    for case in fixture["slices"]:
        actual = _eno3(jnp.array(case["values"]), case["spacing"], extrapolate_away_from_zero)
        for result, key in zip(actual, ["left", "right"], strict=True):
            np.testing.assert_allclose(result, case[key], atol=3e-13, rtol=0)


@pytest.mark.parametrize("method", ["none", "minVWithV0"])
def test_solver_matches_exact_advection_and_streams_a_readable_policy(tmp_path, method):
    from exobalancelab.hjb_solver import solve

    axes = [np.linspace(1.2, 1.9, 7), np.linspace(-1, 1, 5), np.linspace(-2, 2, 7), [0, 0.05, 0.1]]
    _, _, torque = np.meshgrid(*axes[:3], indexing="ij")
    initial = torque
    problem = tmp_path / "problem.mat"
    with h5py.File(problem, "w") as output:
        for key, axis in zip(
            ["theta_rad", "velocity_rad_s", "human_torque_nm", "time_to_go_s"], axes, strict=True
        ):
            output[key] = np.array(axis)[None, :]
        output["state_scale"] = np.ones((1, 3))
        output["time_scale_s"] = [[1]]
        output["rate_bounds"] = [[1, 2]]
        output["initial_value"] = initial.T
        output["obstacle_value"] = np.full(initial.shape[::-1], -100)
        output["drift_theta"] = np.zeros(initial.shape[::-1])
        output["drift_velocity"] = np.zeros(initial.shape[::-1])
        metadata = {"summary": {"method": method}, "contract": policy_contract(PROFILES["OF"], "none")}
        output["problem_json"] = np.array([ord(c) for c in json.dumps(metadata)], dtype=np.uint16)[:, None]
    destination = tmp_path / "solved"
    summary = solve(problem, destination)
    assert summary["stage"] == "solved"
    assert summary["backend"] == "JAX/ENO3"
    with h5py.File(destination / "policy.mat") as policy:
        for i, time in enumerate(axes[3]):
            expected = initial + (time if method == "none" else 0)
            np.testing.assert_allclose(np.asarray(policy["value"][i]).T, expected, atol=1e-12)
    policy = HJBPolicy(destination / "policy.mat")
    policy.validate_plant(PROFILES["OF"], "none")
    assert policy.rate(0.05, 0, 0, 0, PROFILES["OF"]) == PROFILES["OF"].rate_min_nm_s
    with pytest.raises(FileExistsError):
        solve(problem, destination)
