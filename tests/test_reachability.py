"""Lifted HJB execution: torque-state timing and independent ODE comparisons.

Value grids below are analytic interface fixtures, not solved stabilizable regions.
"""

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from exobalancelab.paper import PROFILES
from exobalancelab.reachability import HJBPolicy, PolicyCoverageError, policy_contract
from exobalancelab.simulation import RunConfig, run


def _policy(tmp_path, gradient, horizon_s=1.0):
    torque = np.array([-88.0, 22.0])
    times = np.array([0.0, horizon_s])
    value = np.broadcast_to(torque[None, None, :, None] * np.asarray(gradient), (2, 2, 2, 2))
    path = tmp_path / "linear_value.npz"
    np.savez(
        path,
        theta_rad=[1.2, 1.9],
        velocity_rad_s=[-1, 1],
        human_torque_nm=torque,
        time_to_go_s=times,
        value=value,
        contract_json=json.dumps(policy_contract(PROFILES["OF"], "none")),
    )
    return HJBPolicy(path)


def test_hjb_accepts_time_conversion_roundoff_but_rejects_missing_coverage(tmp_path):
    policy = _policy(tmp_path, 0, horizon_s=np.nextafter(0.03, 0))
    episode = run(RunConfig(duration_s=0.03, lean_rad=0, velocity_rad_s=0), hjb=policy)
    assert episode.metrics["control_calls"] == 30
    with pytest.raises(ValueError, match="time coverage"):
        run(RunConfig(duration_s=0.031), hjb=policy)
    with pytest.raises(ValueError, match="out of bounds"):
        policy.rate(0.03 + 1e-9, 0, 0, 0, PROFILES["OF"])


@pytest.mark.parametrize("gradient,initial_torque", [(1, 0), (-1, 0), (1, -87.5), (-1, 21.8)])
def test_hjb_integrates_held_rate_without_jumping_initial_torque(tmp_path, gradient, initial_torque):
    policy = _policy(tmp_path, gradient)
    config = RunConfig(
        duration_s=0.03, control_period_s=0.01, initial_torque_nm=initial_torque, lean_rad=0, velocity_rad_s=0
    )
    episode = run(config, hjb=policy)
    a = episode.arrays
    p = PROFILES["OF"]
    rate = p.rate_min_nm_s if gradient > 0 else p.rate_max_nm_s
    expected = np.clip(initial_torque + rate * a["time_s"], p.torque_min_nm, p.torque_max_nm)
    np.testing.assert_allclose(a["human_torque_nm"], expected, atol=1e-12)
    np.testing.assert_allclose(a["human_rate_command_nm_s"], rate)
    np.testing.assert_allclose(
        np.diff(a["human_torque_nm"]) / config.timestep_s, a["human_rate_applied_nm_s"][:-1], atol=1e-10
    )
    assert episode.metrics["control_calls"] == 3
    assert a["human_torque_nm"][0] == initial_torque


def test_hjb_samples_rate_only_at_control_ticks(tmp_path):
    # Gradient changes sign at remaining time .015 s, between control ticks.
    policy = _policy(tmp_path, [-0.015, 0.985])
    episode = run(RunConfig(duration_s=0.03, control_period_s=0.01, lean_rad=0, velocity_rad_s=0), hjb=policy)
    p = PROFILES["OF"]
    expected_rate = np.repeat([p.rate_min_nm_s, p.rate_min_nm_s, p.rate_max_nm_s], 10)
    np.testing.assert_allclose(episode.arrays["human_rate_command_nm_s"], expected_rate)
    expected_torque = np.r_[0, np.cumsum(expected_rate[:-1]) * 0.001]
    np.testing.assert_allclose(episode.arrays["human_torque_nm"], expected_torque, atol=1e-12)


def test_lifted_mujoco_trajectory_converges_to_independent_ode(tmp_path):
    policy = _policy(tmp_path, 1)
    p = PROFILES["OF"]

    def derivative(t, state):
        angle, speed, torque = state
        return [
            speed,
            9.81 / p.length_m * np.sin(angle) + (torque - p.damping_nm_s * speed) / p.inertia_kg_m2,
            p.rate_min_nm_s,
        ]

    reference = solve_ivp(derivative, [0, 0.1], [0.01, 0.05, 0], rtol=1e-11, atol=1e-13, dense_output=True)
    errors = []
    for dt in [0.002, 0.001, 0.0005]:
        episode = run(
            RunConfig(
                duration_s=0.1, control_period_s=0.01, timestep_s=dt, lean_rad=0.01, velocity_rad_s=0.05
            ),
            hjb=policy,
        )
        arrays = episode.arrays
        actual = np.array([arrays["lean_rad"], arrays["velocity_rad_s"], arrays["human_torque_nm"]])
        errors.append(np.max(abs(actual - reference.sol(arrays["time_s"]))))
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1]
    assert errors[-1] < 0.0003


def test_initial_torque_respects_scaled_strength():
    with pytest.raises(ValueError, match="strength-scaled"):
        run(RunConfig(human_strength=0.5, initial_torque_nm=20))


@pytest.mark.parametrize("kwargs", [{"parameters": "authors"}, {"exo": "gc"}, {"human_strength": 0.5}])
def test_hjb_rejects_a_different_plant_before_rollout(tmp_path, kwargs):
    policy = _policy(tmp_path, 1)
    with pytest.raises(ValueError, match="HJB"):
        run(RunConfig(duration_s=0.01, **kwargs), hjb=policy)


def test_hjb_requires_parameter_provenance_for_rollout(tmp_path):
    policy = _policy(tmp_path, 1)
    policy.contract = None
    with pytest.raises(ValueError, match="contract_json"):
        run(RunConfig(duration_s=0.01), hjb=policy)


def test_policy_coverage_error_retains_physical_query_and_bounds(tmp_path):
    policy = _policy(tmp_path, 1)
    with pytest.raises(PolicyCoverageError, match="velocity_rad_s") as caught:
        policy.rate(0.5, 0, -1.001, 0, PROFILES["OF"])
    assert caught.value.query["remaining_s"] == 0.5
    assert caught.value.query["velocity_rad_s"] == -1.001
    assert caught.value.bounds["velocity_rad_s"] == [-1, 1]


def test_hdf5_policy_matches_full_interpolation_without_loading_the_history(tmp_path, monkeypatch):
    axes = ([1.2, 1.5, 1.9], [-1, -0.1, 0.3, 1], [-88, -30, -3, 0, 22], [0, 0.01, 0.04, 0.1])
    rng = np.random.default_rng(240805418)
    value = rng.normal(size=tuple(map(len, axes)))
    path = tmp_path / "policy.mat"
    with h5py.File(path, "w") as target:
        for key, axis in zip(
            ["theta_rad", "velocity_rad_s", "human_torque_nm", "time_to_go_s"], axes, strict=True
        ):
            target[key] = np.asarray(axis)[None, :]
        target["value"] = value.T
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    original_array = h5py.Dataset.__array__

    def bounded_array(dataset, *args, **kwargs):
        assert dataset.name != "/value", "The full four-dimensional history must stay on disk"
        return original_array(dataset, *args, **kwargs)

    def no_read_bytes(path):
        raise AssertionError("Policy hashing must stream rather than allocate the entire file")

    monkeypatch.setattr(h5py.Dataset, "__array__", bounded_array)
    monkeypatch.setattr(Path, "read_bytes", no_read_bytes)
    policy = HJBPolicy(path)
    cache = tmp_path / "gradient_cache"
    cached = HJBPolicy(path, gradient_cache=cache)
    reused = HJBPolicy(path, gradient_cache=cache)
    reference = RegularGridInterpolator(axes, np.gradient(value, axes[2], axis=2, edge_order=1))
    # Reverse and revisit time slices, including endpoints and nonuniform knots.
    for time in [0.1, 0.08, 0.04, 0.015, 0.01, 0, 0.1, 0.005]:
        for _ in range(10):
            point = [rng.uniform(axis[0], axis[-1]) for axis in axes[:3]] + [time]
            np.testing.assert_allclose(policy.gradient(point), reference(point), rtol=1e-13, atol=1e-14)
            np.testing.assert_array_equal(cached.gradient(point), policy.gradient(point))
            np.testing.assert_array_equal(reused.gradient(point), policy.gradient(point))
            assert len(policy.gradient._slices) <= 2
    assert policy.sha256 == expected_hash
    assert policy.sha256 == expected_hash
    with pytest.raises(PolicyCoverageError):
        policy.rate(0.101, 0, 0, 0, PROFILES["OF"])
    different = tmp_path / "different.mat"
    shutil.copyfile(path, different)
    with h5py.File(different, "a") as target:
        target["value"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="different HJB policy"):
        HJBPolicy(different, gradient_cache=cache)
    with (cache / "gradient.npy").open("r+b") as target:
        target.seek(-1, 2)
        previous = target.read(1)[0]
        target.seek(-1, 2)
        target.write(bytes([previous ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        HJBPolicy(path, gradient_cache=cache)
    # Invalid values must still be rejected before a rollout, even in the final slice.
    with h5py.File(path, "a") as target:
        target["value"][-1, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        HJBPolicy(path)
