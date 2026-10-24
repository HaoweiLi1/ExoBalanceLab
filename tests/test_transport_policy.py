"""Moving-grid policy coordinates, native decisions, goal choice and plant contracts."""

import json

import h5py
import numpy as np
import pytest

from exobalancelab.paper import AUTHORS_PROFILES
from exobalancelab.reachability import PolicyCoverageError, policy_contract
from exobalancelab.transport_policy import FORMAT, TransportPolicy, switching_codes


def fixture(path, complete=True):
    p = AUTHORS_PROFILES["OM"]
    with h5py.File(path, "w") as f:
        f.attrs.update(format=FORMAT, complete=complete, contract_json=json.dumps(policy_contract(p, "none")))
        f["time_to_go_s"] = [0, 0.5, 1]
        f["rate_bounds_nm_s"] = [p.rate_min_nm_s, p.rate_max_nm_s]
        for goal in range(2):
            g = f.create_group(f"goals/{goal}")
            g["theta_rad"] = [[1, 1.5, 2], [0.9, 1.5, 2.1], [0.8, 1.5, 2.2]]
            g["velocity_rad_s"] = [[-1, 0, 1], [-2, 0, 2], [-3, 0, 3]]
            g["human_torque_nm"] = [[-1, 0, 1], [-2, 0, 2], [-3, 0, 3]]
            g["final_value"] = np.full((3, 3, 3), -1.0 if goal == 0 else 1.0)
            for k in range(3):
                a = np.ones((3, 3, 3), dtype=np.int8)
                if goal == 0:
                    a[:, :, :] = np.array([1, 0, -1]) * (1 if k < 2 else -1)
                g[f"controls/{k:04d}"] = a
    return p


def test_native_derivative_decisions_preserve_zero_and_both_signs():
    value = np.broadcast_to(np.array([4.0, 1.0, 0.0, 1.0, 4.0]), (2, 3, 5))
    codes = switching_codes(value)
    np.testing.assert_array_equal(codes, np.broadcast_to([-1, -1, 0, 1, 1], value.shape))
    assert codes.dtype == np.int8


def test_moving_axes_time_and_rate_interpolation_with_fixed_goal(tmp_path):
    path = tmp_path / "policy.h5"
    p = fixture(path)
    with TransportPolicy(path) as policy:
        policy.validate_plant(p, "none")
        np.testing.assert_array_equal(policy.initial_values([1.5, 0, 1]), [-1, 1])
        q = 1.5 - np.pi / 2
        assert policy.rate(1, q, 0, 1, p) == pytest.approx(p.rate_min_nm_s / 3)
        assert policy.selected_goal == 0
        expected = 0.2 * (p.rate_max_nm_s + p.rate_min_nm_s)
        assert policy.rate(0.75, q, 0, 1, p) == pytest.approx(expected)
        assert policy.rate(0.5, q, 0, 1, p) == pytest.approx(p.rate_max_nm_s / 2)
        # At this knot, the state belongs to the 0.5 s grid but not the 0 s grid.
        assert policy.rate(np.nextafter(0.5, 0), q, 0, 1.5, p) == pytest.approx(0.75 * p.rate_max_nm_s)
        assert policy.rate(0, q, 0, 0, p) == 0
        assert len(policy._controls) == 2
        # This is inside the interpolated mesh, although outside the lower-time mesh.
        assert policy.rate(0.25, q, 0, 1.5, p) == p.rate_max_nm_s
        with pytest.raises(PolicyCoverageError):
            policy.rate(0.25, q, 0, 1.500001, p)
        with pytest.raises(ValueError, match="does not match"):
            policy.validate_plant(p, "gc")


def test_partial_export_is_not_a_policy(tmp_path):
    path = tmp_path / "policy.h5"
    fixture(path, False)
    with pytest.raises(ValueError, match="Incomplete"):
        TransportPolicy(path)


def test_cli_loads_native_policy_and_preserves_initial_torque(tmp_path, monkeypatch):
    from exobalancelab.cli import main
    from exobalancelab.reachability import load_policy
    from exobalancelab.simulation import RunConfig, run

    path = tmp_path / "policy.h5"
    fixture(path)
    with h5py.File(path, "r+") as f:
        for goal in f["goals"].values():
            for codes in goal["controls"].values():
                codes[:] = 0
    with load_policy(path) as policy:
        with pytest.raises(ValueError, match="supported duration"):
            run(RunConfig(profile="OM", parameters="authors", duration_s=0.5), hjb=policy)
    output = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "exobalance",
            "run",
            "--profile",
            "OM",
            "--parameters",
            "authors",
            "--hjb-policy",
            str(path),
            "--duration",
            "1",
            "--lean",
            "0",
            "--velocity",
            "0",
            "--initial-torque",
            "0.5",
            "--out",
            str(output),
        ],
    )
    main()
    with np.load(output / "none/episode.npz") as episode:
        np.testing.assert_allclose(episode["human_torque_nm"], 0.5)
    assert json.loads((output / "none/config.json").read_text())["initial_torque_nm"] == 0.5
