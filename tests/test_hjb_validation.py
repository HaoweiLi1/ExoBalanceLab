"""Frozen HJB validation cohorts and inspectable coverage failures in SI units."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from exobalancelab.hjb_validation import validate
from exobalancelab.paper import AUTHORS_PROFILES
from exobalancelab.reachability import policy_contract


@pytest.mark.parametrize("speed,torque,expected", [(0, 0, "completed"), (0.099, 22, "rollout_error")])
def test_frozen_states_are_reused_and_failures_retain_the_query(
    tmp_path, monkeypatch, speed, torque, expected
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
        contract = json.dumps(policy_contract(AUTHORS_PROFILES["OF"], "none"))
        f["contract_json"] = np.array([ord(c) for c in contract], dtype=np.uint16)[:, None]
    target = {}
    for label in ["Ank", "Toe"]:
        target.update({f"theta{label}Range": 0.1, f"thetaDot{label}Range": 0.1, f"torque{label}Range": 30})
    target.update(ankleEq=[np.pi / 2, 0, 0], toeEq=[np.pi / 2, 0, 0])
    (source / "summary.json").write_text(json.dumps({"profile": "OF", "target": target}))
    state = [np.pi / 2, speed, torque]
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
    result = validate(source, output, seed=999, protocol_path=original)
    assert result["selected"] == 1
    case = result["outcomes"][0]
    assert case["state"] == state
    assert case["group"] == "inside"
    assert case["selection_value"] == -5
    assert case["value"] == 1
    assert not case["predicted_inside"]
    assert case["status"] == expected
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["seed"] == 123
    assert protocol["requested_per_group"] == 1
    assert protocol["selection_protocol_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    if expected == "rollout_error":
        assert 0 < case["elapsed_s"] < 0.01
        assert case["failed_query"]["velocity_rad_s"] > case["policy_bounds"]["velocity_rad_s"][1]
    else:
        assert case["valid_target_reach"]
        assert (output / "episode_00" / "episode.npz").exists()
