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


def test_godunov_control_matches_concave_hamiltonian_extrema():
    import jax.numpy as jnp

    from exobalancelab.hjb_solver import _godunov_control

    left, right = np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-2, 2, 9))
    for bounds in [(-2, 1), (1, 2), (-2, -1), (0, 0)]:
        points = np.linspace(left, right, 1001)
        hamiltonian = np.minimum(bounds[0] * points, bounds[1] * points)
        expected = np.where(left <= right, hamiltonian.max(axis=0), hamiltonian.min(axis=0))
        # Include the analytic corner p=0 for extrema between sampled points.
        expected = np.where((left <= 0) & (right >= 0), np.maximum(expected, 0), expected)
        actual = _godunov_control(jnp.array(left), jnp.array(right), jnp.array(bounds))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_godunov_retains_underresolved_terminal_interval():
    import hj_reachability as hj
    import jax.numpy as jnp

    from exobalancelab.hjb_solver import advance

    # Only one negative node: the exact solution retains its negative depth
    # while the controllable interval expands asymmetrically under [-2, 1].
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-1.0, -1.0, -1.0]), jnp.ones(3)), (3, 3, 401)
    )
    torque = np.asarray(grid.states[..., 2])
    initial = jnp.abs(grid.states[..., 2]) - 0.002
    value = advance(
        initial,
        0.0,
        0.25,
        grid,
        jnp.zeros_like(grid.states),
        jnp.full(initial.shape, -100.0),
        jnp.array([-2.0, 1.0]),
        initial,
        False,
        flux="godunov",
    )
    exact = np.maximum.reduce([-torque - 0.25, torque - 0.5, np.zeros_like(torque)]) - 0.002
    # Check the interior, away from the two propagating nonsmooth corners.
    interior = (torque > -0.2) & (torque < 0.4)
    np.testing.assert_allclose(np.asarray(value)[interior], exact[interior], atol=3e-5, rtol=0)
    assert np.min(value) >= np.min(initial) - 1e-14
    assert np.all(np.asarray(value)[interior] < 0)


def test_mapped_grid_uses_coordinate_jacobian_and_converges_to_physical_advection(tmp_path):
    from exobalancelab.hjb_solver import solve

    errors = []
    for n in [41, 81]:
        coordinate = np.linspace(-2, 2, n)
        theta = np.pi / 2 + 0.1 * np.sinh(coordinate)
        axes = [theta, np.linspace(-1, 1, 5), np.linspace(-2, 2, 7), [0, 0.05]]
        initial = np.broadcast_to(theta[:, None, None], (n, 5, 7))
        velocity = np.broadcast_to((2 / np.cosh(coordinate))[:, None, None], initial.shape)
        problem = tmp_path / f"mapped{n}.mat"
        with h5py.File(problem, "w") as output:
            for key, axis in zip(["theta_rad", "velocity_rad_s", "human_torque_nm", "time_to_go_s"], axes):
                output[key] = np.array(axis)[None, :]
            for key, axis in zip(
                ["solver_theta", "solver_velocity", "solver_torque"], [coordinate, axes[1], axes[2]]
            ):
                output[key] = np.array(axis)[None, :]
            output["state_scale"] = np.ones((1, 3))
            output["time_scale_s"] = [[1]]
            output["rate_bounds"] = [[0, 0]]
            output["initial_value"] = initial.T
            output["drift_theta"] = velocity.T
            output["drift_velocity"] = np.zeros(initial.shape[::-1])
            output["obstacle_value"] = np.full(initial.shape[::-1], -100)
            metadata = {"summary": {"method": "none"}, "contract": {}}
            output["problem_json"] = np.array([ord(c) for c in json.dumps(metadata)], dtype=np.uint16)[
                :, None
            ]
        destination = tmp_path / f"solved{n}"
        result = solve(problem, destination, save_history=False, platform="cpu", flux="godunov")
        assert result["coordinate_grid"] == "mapped"
        actual = np.load(destination / "final_value.npy")
        interior = abs(coordinate) < 1
        errors.append(np.max(abs(actual[interior] - (initial + 0.01)[interior])))
    assert errors[0] < 2e-5
    assert errors[1] < errors[0] / 4


@pytest.mark.parametrize("method", ["none", "minVWithV0"])
@pytest.mark.parametrize("flux", ["lax_friedrichs", "godunov"])
def test_solver_matches_exact_advection_and_streams_a_readable_policy(tmp_path, monkeypatch, method, flux):
    from exobalancelab import hjb_solver
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
    summary = solve(problem, destination, flux=flux)
    assert summary["stage"] == "solved"
    assert summary["backend"] == "JAX/ENO3"
    assert not (destination / "policy.partial.mat").exists()
    with h5py.File(destination / "policy.mat") as policy:
        for i, time in enumerate(axes[3]):
            expected = initial + (time if method == "none" else 0)
            np.testing.assert_allclose(np.asarray(policy["value"][i]).T, expected, atol=1e-12)
    policy = HJBPolicy(destination / "policy.mat")
    policy.validate_plant(PROFILES["OF"], "none")
    assert policy.rate(0.05, 0, 0, 0, PROFILES["OF"]) == PROFILES["OF"].rate_min_nm_s
    with pytest.raises(FileExistsError):
        solve(problem, destination, flux=flux)
    if method == "none":

        def failed_export(*args, **kwargs):
            raise OSError("Injected final export failure")

        monkeypatch.setattr(hjb_solver, "savemat", failed_export)
        interrupted = tmp_path / "interrupted"
        with pytest.raises(OSError, match="export failure"):
            solve(problem, interrupted, flux=flux)
        assert (interrupted / "policy.partial.mat").exists()
        assert not (interrupted / "policy.mat").exists()
        assert json.loads((interrupted / "summary.json").read_text())["stage"] == "running"
