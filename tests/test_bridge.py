"""Controller transport contracts: units, coverage, actual samples and errors."""

import json

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from exobalancelab.bridge import Trajectory
from exobalancelab.simulation import RunConfig, run


@pytest.mark.parametrize("format", ["csv", "legacy", "mat", "hdf"])
def test_matlab_formats_preserve_time_and_channels(tmp_path, format):
    times = np.array([0, 0.1, 0.4])
    values = np.array([[1, 2], [3, 4], [9, 10]])
    path = tmp_path / ("input.mat" if format in {"mat", "hdf"} else "input.csv")
    if format == "mat":
        savemat(path, {"time_s": times, "torque_nm": values})
    elif format == "hdf":
        with h5py.File(path, "w") as f:
            f["time_s"] = times[None, :]
            f["torque_nm"] = values.T
    else:
        np.savetxt(
            path,
            np.column_stack([times, values]),
            delimiter=",",
            header="time_s,left_nm,right_nm" if format == "csv" else "",
            comments="",
        )
    trajectory = Trajectory.load(path)
    np.testing.assert_allclose(trajectory.sample(0.25), [6, 7])
    np.testing.assert_allclose(trajectory.sample(0), [1, 2])
    np.testing.assert_allclose(trajectory.sample(0.4), [9, 10])
    with pytest.raises(ValueError, match="coverage"):
        trajectory.sample(0.401)


@pytest.mark.parametrize(
    "times,values", [([0, 0], [1, 2]), ([0, -1], [1, 2]), ([0, 1], [1, np.nan]), ([0], [1]), ([0, 1], [[1]])]
)
def test_invalid_samples_fail(times, values):
    with pytest.raises(ValueError):
        Trajectory(times, values)


def test_missing_file_is_not_zero_torque(tmp_path):
    with pytest.raises(FileNotFoundError):
        Trajectory.load(tmp_path / "missing.csv")


def test_zero_order_hold_and_explicit_endpoint_hold():
    traj = Trajectory([0, 0.1, 0.2], [0, 2, 3], "zoh", "hold")
    assert traj.sample(0.099) == 0
    assert traj.sample(0.1) == 2
    assert traj.sample(5) == 3


def test_replay_drives_mujoco_at_current_time_without_lookahead():
    trajectory = Trajectory([0, 0.1, 0.2], [0, 1, 2])
    episode = run(RunConfig(duration_s=0.2, lean_rad=0, velocity_rad_s=0), trajectory)
    a = episode.arrays
    np.testing.assert_allclose(a["human_raw_nm"][:, 0], 10 * a["time_s"], atol=1e-12)
    assert a["qvel"][-1, 0] > 0


def test_multirate_control_holds_commands_between_ticks():
    trajectory = Trajectory([0, 0.1], [0, 1])
    episode = run(RunConfig(duration_s=0.1, control_period_s=0.005), trajectory)
    expected = np.repeat(np.arange(20) * 0.05, 5)
    np.testing.assert_allclose(episode.arrays["human_raw_nm"][:, 0], expected, atol=1e-12)
    assert episode.metrics["control_calls"] == 20


def test_external_channels_must_match_model():
    with pytest.raises(ValueError, match="channels"):
        run(RunConfig(duration_s=0.1), Trajectory([0, 1], [[0, 1], [1, 2]]))


def test_hjb_torque_rate_sign_and_grid_coverage(tmp_path):
    from exobalancelab.paper import PROFILES
    from exobalancelab.reachability import HJBPolicy, policy_contract

    path = tmp_path / "policy.npz"
    theta = np.array([1.2, 1.9])
    velocity = np.array([-1, 1])
    torque = np.array([-88, 22])
    times = np.array([0, 1])
    # V = speed * human_torque. This is an interface fixture, not a solved BRS.
    value = np.broadcast_to(velocity[None, :, None, None] * torque[None, None, :, None], (2, 2, 2, 2))
    np.savez(
        path,
        theta_rad=theta,
        velocity_rad_s=velocity,
        human_torque_nm=torque,
        time_to_go_s=times,
        value=value,
        contract_json=json.dumps(policy_contract(PROFILES["OF"], "none")),
    )
    policy = HJBPolicy(path)
    p = PROFILES["OF"]
    assert policy.rate(0.5, 0, 0.5, 0, p) == p.rate_min_nm_s
    assert policy.rate(0.5, 0, -0.5, 0, p) == p.rate_max_nm_s
    with pytest.raises(ValueError):
        policy.rate(0.5, 0, 2, 0, p)
    episode = run(RunConfig(duration_s=0.01, velocity_rad_s=0.1, lean_rad=0), hjb=policy)
    assert episode.arrays["human_applied_nm"][0, 0] == 0
    assert episode.arrays["human_applied_nm"][1, 0] < 0
    assert episode.metrics["controller"] == "hjb"


def test_legacy_entrypoints_share_validated_transport(tmp_path):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    path = tmp_path / "controller.csv"
    np.savetxt(path, [[0, 0], [0.01, 1]], delimiter=",")
    for directory in ["joint1_humanoid", "joint2_humanoid"]:
        module_path = Path(__file__).resolve().parents[1] / directory / "controllers.py"
        spec = importlib.util.spec_from_file_location(directory + "_controllers", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = SimpleNamespace(opt=SimpleNamespace(timestep=0.001))
        data = SimpleNamespace(time=0.005)
        controller = module.HumanPrecomputedController(model, data, {"trajectory_file": str(path)})
        assert controller.compute_control(None, None) == pytest.approx(0.5)
        data.time = 0.1
        with pytest.raises(ValueError, match="coverage"):
            controller.compute_control(None, None)
