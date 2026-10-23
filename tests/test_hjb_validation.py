"""Frozen HJB validation cohorts and inspectable coverage failures in SI units."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from exobalancelab.hjb_validation import validate
from exobalancelab.paper import AUTHORS_PROFILES
from exobalancelab.reachability import policy_contract


@pytest.mark.parametrize(
    "speed,torque,expected,assistance",
    [
        (0, 0, "completed", "none"),
        (0.099, 22, "rollout_error", "none"),
        (0, 0, "completed", "gc"),
        (0, 0, "completed", "pd"),
    ],
)
def test_frozen_states_are_reused_and_failures_retain_the_query(
    tmp_path, monkeypatch, speed, torque, expected, assistance
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    source = tmp_path / "solve"
    source.mkdir()
    with h5py.File(source / "policy.mat", "w") as f:
        for key, values in {
            "theta_rad": [1.4, 1.8],
            "velocity_rad_s": [-0.1, 0.1],
            "human_torque_nm": [-88, 22],
            "time_to_go_s": [0, 0.01],
        }.items():
            f[key] = np.array(values)[:, None]
        # Constant synthetic cost: zero optimal rate; not a solved HJB field.
        f["value"] = np.ones((2, 2, 2, 2))
        contract = json.dumps(policy_contract(AUTHORS_PROFILES["OF"], assistance))
        f["contract_json"] = np.array([ord(c) for c in contract], dtype=np.uint16)[:, None]
    target = {}
    for label in ["Ank", "Toe"]:
        target.update({f"theta{label}Range": 0.1, f"thetaDot{label}Range": 0.1, f"torque{label}Range": 30})
    target.update(ankleEq=[np.pi / 2, 0, 0], toeEq=[np.pi / 2, 0, 0])
    (source / "summary.json").write_text(json.dumps({"profile": "OF", "target": target, "exo": assistance}))
    state = [np.pi / 2 + (0.01 if assistance != "none" else 0), speed, torque]
    original = tmp_path / "selection.json"
    original.write_text(
        json.dumps(
            {
                "seed": 123,
                "requested_per_group": 1,
                "policy_sha256": "original-policy",
                "states": [{"group": "inside", "state": state, "value": -5}],
            }
        )
    )
    output = tmp_path / "validation"
    timestep = 0.0005 if assistance == "pd" else 0.001
    result = validate(
        source,
        output,
        seed=999,
        protocol_path=original,
        timestep_s=timestep,
        control_period_s=timestep,
        gradient_cache=tmp_path / "cache" if assistance == "pd" else None,
    )
    assert result["selected"] == 1
    case = result["outcomes"][0]
    assert case["state"] == state
    assert case["group"] == "inside"
    assert case["selection_value"] == -5
    assert case["value"] == pytest.approx(1, rel=0, abs=1e-14)
    assert not case["predicted_inside"]
    assert case["status"] == expected
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["seed"] == 123
    assert protocol["requested_per_group"] == 1
    assert protocol["rollout"] == {
        "timestep_s": timestep,
        "control_period_s": timestep,
        "command_delay_s": 0.0,
    }
    if assistance == "pd":
        assert protocol["gradient_cache"]["policy_sha256"] == protocol["policy_sha256"]
    assert protocol["selection_protocol_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    if expected == "rollout_error":
        assert 0 < case["elapsed_s"] < 0.01
        assert case["failed_query"]["velocity_rad_s"] > case["policy_bounds"]["velocity_rad_s"][1]
    else:
        assert case["valid_target_reach"]
        assert (output / "episode_00" / "episode.npz").exists()
        metrics = json.loads((output / "episode_00" / "metrics.json").read_text())
        assert metrics["physics_hz"] == metrics["control_hz"] == 1 / timestep
        if assistance != "none":
            with np.load(output / "episode_00" / "episode.npz") as episode:
                assert np.all(episode["exo_torque_nm"] < 0)


def test_new_cohorts_respect_strength_limits_but_frozen_invalid_states_remain_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    source = tmp_path / "solve"
    source.mkdir()
    profile = AUTHORS_PROFILES["OF"]
    with h5py.File(source / "policy.mat", "w") as file:
        for key, values in {
            "theta_rad": [np.pi / 2, 1.819375],
            "velocity_rad_s": [0, 0.01],
            "human_torque_nm": [-90, -88, 0, 22, 24],
            "time_to_go_s": [0, 0.01],
        }.items():
            file[key] = np.array(values)[:, None]
        file["value"] = np.ones((2, 5, 2, 2))
        contract = json.dumps(policy_contract(profile, "none"))
        file["contract_json"] = np.array([ord(c) for c in contract], dtype=np.uint16)[:, None]
    target = {"ankleEq": [np.pi / 2, 0, 0], "toeEq": [np.pi / 2, 0, 0]}
    for label in ["Ank", "Toe"]:
        target.update({f"theta{label}Range": 0.1, f"thetaDot{label}Range": 0.1, f"torque{label}Range": 30})
    (source / "summary.json").write_text(json.dumps({"profile": "OF", "exo": "none", "target": target}))
    output = tmp_path / "automatic"
    result = validate(source, output, count=100)
    torques = {case["state"][2] for case in result["outcomes"]}
    assert torques == {-88, 0, 22}
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["selection_constraints"]["human_torque_nm"] == [-88, 22]

    # Reusing an old protocol must preserve its invalid initial state and failure.
    protocol["states"] = [{"group": "outside", "state": [np.pi / 2, 0, 24], "value": 1}]
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(protocol))
    result = validate(source, tmp_path / "replay", protocol_path=frozen)
    assert result["outcomes"][0]["state"][2] == 24
    assert result["outcomes"][0]["status"] == "rollout_error"
    assert "Initial human torque exceeds" in result["outcomes"][0]["error"]
