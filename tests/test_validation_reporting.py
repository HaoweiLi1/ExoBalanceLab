"""Incomplete cohorts must not turn into apparently improved recovery percentages."""

import importlib.util
import json
from pathlib import Path

import pytest


def _report():
    path = Path(__file__).resolve().parents[1] / "examples/report_validation.py"
    spec = importlib.util.spec_from_file_location("validation_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_report_rejects_missing_failed_triplet(tmp_path):
    directory = tmp_path / "paired"
    directory.mkdir()
    (directory / "protocol.json").write_text(json.dumps({"cases": [{"id": "failing_pair_none"}]}))
    (directory / "outcomes.json").write_text("[]")
    with pytest.raises(ValueError, match="every frozen case"):
        _report().paired_report(tmp_path, tmp_path)


def test_timing_report_rejects_missing_coverage_error(tmp_path):
    directory = tmp_path / "timing"
    directory.mkdir()
    setting = dict(
        physics_dt_s=0.001,
        physics_reference_dt_s=0.0005,
        physics_reference_control_period_s=0.001,
        control_periods_s=[0.001],
        command_delays_s=[0],
    )
    protocol = {
        "parent_protocol": {"timing": setting},
        "groups": [{"id": "OF_gc_forward", "states": [{"state": [1.5, 0.1, 0]}]}],
    }
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "outcomes.json").write_text("[]")
    with pytest.raises(ValueError, match="every frozen case"):
        _report().timing_report(tmp_path, tmp_path, {})


def test_timing_refinement_cannot_hide_changed_contact_behind_combined_failure(tmp_path):
    directory = tmp_path / "timing"
    directory.mkdir()
    setting = dict(
        physics_dt_s=0.0000125,
        physics_reference_dt_s=0.00000625,
        physics_reference_control_period_s=0.001,
        control_periods_s=[0.001],
        command_delays_s=[0],
    )
    state = {"state": [1.57, 0.001, 0], "group": "boundary"}
    protocol = {
        "parent_protocol": {"timing": setting},
        "groups": [{"id": "OF_gc_forward", "states": [state]}],
    }
    results = []
    for dt, target, contact in [(0.0000125, False, True), (0.00000625, True, False)]:
        results.append(
            dict(
                id=str(dt),
                policy_id="OF_gc_forward",
                state=state,
                config=dict(timestep_s=dt, control_period_s=0.001, command_delay_s=0),
                predicted_inside=False,
                status="completed",
                final_state=state["state"],
                target_reached=target,
                contact_valid=contact,
                valid_target_reach=False,
            )
        )
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "outcomes.json").write_text(json.dumps(results))
    acceptance = {"rollout_endpoint_abs": {"angle_rad": 0.001, "velocity_rad_s": 0.01, "torque_nm": 1}}
    summary = _report().timing_report(tmp_path, tmp_path, acceptance)
    assert summary["physics_refinement_evaluable"] == 1
    assert summary["physics_refinement_passed"] == 0
    assessment = json.loads((tmp_path / "timing_refinement.json").read_text())[0]
    assert assessment["assessment"] == "fail"
    assert not assessment["target_and_contact_classifications_equal"]
