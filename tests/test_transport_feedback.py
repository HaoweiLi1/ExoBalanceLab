"""Feedback report rejects endpoint, classification, coverage and deadline failures."""

import copy
import json

from examples.validate_transport_feedback import report, specification, write


def test_feedback_report_preserves_original_limits_and_counts_failed_runs(tmp_path):
    spec = specification()
    write(tmp_path / "protocol.json", spec)
    write(tmp_path / "selection_OM.json", {"samples": [{"index": 0, "group": "inside"}]})
    base = dict(
        status="completed",
        predicted_inside=True,
        final_state=[1.57, 0, 0],
        target_reached=True,
        contact_valid=True,
        valid_target_reach=True,
        metrics={"control_p99_us": 999},
    )

    def save(mode, step, result):
        write(tmp_path / "episodes" / f"OM_{mode}_00_dt{step:g}" / "result.json", result)

    for mode in ["none", "gc"]:
        for dt in spec["physics_steps_s"]:
            save(mode, dt, base)
    assert report(tmp_path, ["OM"])["all_refinements_passed"]
    bad = copy.deepcopy(base)
    bad["final_state"][0] += 0.00101
    bad["metrics"]["control_p99_us"] = 1001
    save("none", spec["physics_steps_s"][1], bad)
    bad = copy.deepcopy(base)
    bad["contact_valid"] = False
    bad["valid_target_reach"] = False
    save("gc", spec["physics_steps_s"][1], bad)
    result = report(tmp_path, ["OM"])
    assert result["refinement_passed"] == 0
    assert result["deadline_passed"] == 3
    save("gc", spec["physics_steps_s"][1], dict(status="rollout_error", predicted_inside=True))
    result = report(tmp_path, ["OM"])
    assert result["completed"] == 3
    assert result["predicted_inside"] == 4
    assert result["inside_valid_reach"] == 3
    assert result["refinement_passed"] == 0
    assert json.loads((tmp_path / "summary.json").read_text()) == result
