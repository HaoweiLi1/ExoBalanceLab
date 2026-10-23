"""Independent strength limits, causal delay and physically paired initialization."""

from dataclasses import replace

import numpy as np
import pytest

from exobalancelab.bridge import Trajectory
from exobalancelab.paper import PROFILES, scale_strength
from exobalancelab.simulation import RunConfig, run


def test_independent_strength_limits_reach_simulator():
    p = PROFILES["OF"]
    mt, rate = scale_strength(p, 0.5, 1), scale_strength(p, 1, 0.5)
    assert mt.rate_min_nm_s == p.rate_min_nm_s
    assert rate.torque_min_nm == p.torque_min_nm
    command = Trajectory([0, 1], [-1000, -1000])
    config = RunConfig(duration_s=0.5, lean_rad=0, velocity_rad_s=0, control_period_s=0.01)
    a = run(replace(config, mt_scale=0.5), trajectory=command).arrays["human_torque_nm"]
    b = run(replace(config, mrtd_scale=0.5), trajectory=command).arrays["human_torque_nm"]
    assert a[0] == pytest.approx(p.rate_min_nm_s * config.timestep_s)
    assert b[0] == pytest.approx(p.rate_min_nm_s * 0.5 * config.timestep_s)
    assert min(a) == p.torque_min_nm * 0.5
    assert min(b) == p.torque_min_nm


def test_delayed_replay_cannot_affect_plant_before_command_arrival():
    command = Trajectory([0, 0.1], [1000, 1000])
    config = RunConfig(
        duration_s=0.05, control_period_s=0.01, command_delay_s=0.015, lean_rad=0, velocity_rad_s=0
    )
    episode = run(config, trajectory=command)
    a = episode.arrays
    np.testing.assert_array_equal(a["human_torque_nm"][:15], 0)
    assert a["human_torque_nm"][15] > 0
    np.testing.assert_allclose(a["command_sample_time_s"][15:25], 0)
    assert a["command_sample_time_s"][25] == 0.01
    # Even at a slow command rate, the actuator cannot jump by a full period.
    assert max(np.diff(a["human_torque_nm"])) <= PROFILES["OF"].rate_max_nm_s * 0.001 + 1e-12


def test_terminal_contact_uses_delivered_device_torque():
    import importlib.util
    from pathlib import Path

    from exobalancelab.simulation import Episode

    spec = importlib.util.spec_from_file_location(
        "validation_rollouts", Path(__file__).resolve().parents[1] / "examples/validation_rollouts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Current GC would be negative. A queued positive torque is still applied and
    # moves CoP behind the heel, despite the state lying inside the terminal box.
    target = {"ankleEq": [np.pi / 2 + 0.03, 0, 0], "toeEq": [np.pi / 2 + 0.03, 0, 0]}
    for label in ["Ank", "Toe"]:
        for prefix in ["theta", "thetaDot", "torque"]:
            target[prefix + label + "Range"] = 0.01
    metrics = dict(
        final_lean_rad=0.03, final_velocity_rad_s=0, final_human_torque_nm=0, contact_valid_fraction=1
    )
    episode = Episode(RunConfig(exo="gc"), "", {"exo_torque_nm": np.array([50.0])}, metrics)
    outcome = module.target_outcome(episode, dict(profile="OF", exo="gc", target=target))
    assert outcome["target_reached"]
    assert not outcome["contact_valid"]
    assert not outcome["valid_target_reach"]


def test_human_pair_keeps_initial_contacts_and_fixed_feedback_identical():
    config = RunConfig(
        model="human",
        duration_s=0.02,
        lean_rad=0.06,
        velocity_rad_s=0.3,
        initial_posture="flat_feet",
        human_feedback="fixed",
        control_period_s=0.005,
    )
    none = run(config)
    gc = run(replace(config, exo="gc"))
    np.testing.assert_array_equal(none.metrics["initial_state"], gc.metrics["initial_state"])
    np.testing.assert_array_equal(none.metrics["feedback_gain"], gc.metrics["feedback_gain"])
    assert none.arrays["foot_pitch_rad"][0] < 0.001
    assert none.arrays["normal_n"][0] > 400
    adapted = run(replace(config, exo="gc", human_feedback="readapted"))
    np.testing.assert_array_equal(none.metrics["initial_state"], adapted.metrics["initial_state"])
    assert not np.allclose(none.metrics["feedback_gain"], adapted.metrics["feedback_gain"])
    refined = run(replace(config, exo="gc", human_feedback="readapted", timestep_s=0.0005))
    np.testing.assert_array_equal(adapted.metrics["initial_state"], refined.metrics["initial_state"])
    np.testing.assert_array_equal(adapted.metrics["feedback_gain"], refined.metrics["feedback_gain"])


@pytest.mark.parametrize(
    "kwargs", [{"mt_scale": 0}, {"mrtd_scale": 1.1}, {"command_delay_s": -0.01}, {"command_delay_s": 0.0005}]
)
def test_invalid_strength_or_delay_rejected(kwargs):
    with pytest.raises(ValueError):
        run(RunConfig(duration_s=0.01, **kwargs))
