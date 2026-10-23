"""Paired numerical reporting retains missed targets and incomplete rollouts."""

import json

import pytest

from exobalancelab.hjb_comparison import compare


@pytest.fixture
def runs(tmp_path):
    states = [
        {"group": group, "state": [1.5, 0.1 * index, -10.0], "value": -0.1}
        for index, group in enumerate(["inside", "outside", "boundary"])
    ]
    protocol = {
        "states": states,
        "source": {
            "profile": "OF",
            "source_model": {"m": 60},
            "gravity_m_s2": 9.8,
            "exo": "none",
            "target": {"width": 0.01},
            "duration_s": 1.4,
        },
        "policy_sha256": "fixture",
    }
    outcomes = [
        dict(states[0], predicted_inside=True, status="completed", valid_target_reach=True),
        dict(states[1], predicted_inside=False, status="completed", valid_target_reach=False),
        dict(states[2], predicted_inside=True, status="rollout_error", error="out of grid"),
    ]
    result = {}
    for label in ["coarse", "fine"]:
        directory = tmp_path / label
        directory.mkdir()
        (directory / "protocol.json").write_text(json.dumps(protocol))
        (directory / "outcomes.json").write_text(json.dumps(outcomes))
        result[label] = directory
    return result


def test_comparison_keeps_every_outcome(runs, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output = tmp_path / "comparison"
    summaries = compare(runs, output)
    for row in summaries:
        assert row["total"] == 3
        assert row["valid_target"] == row["completed_miss"] == row["policy_error"] == 1
        assert row["predicted_inside"] == 2
        assert row["predicted_inside_reached"] == 1
    assert len((output / "cases.csv").read_text().splitlines()) == 7
    assert (output / "outcomes.pdf").exists()


@pytest.mark.parametrize("change", ["drop_failure", "target", "state", "gravity"])
def test_changed_cohorts_and_missing_failures_are_rejected(runs, tmp_path, change):
    path = runs["fine"] / ("outcomes.json" if change == "drop_failure" else "protocol.json")
    data = json.loads(path.read_text())
    if change == "drop_failure":
        data.pop()
    elif change == "target":
        data["source"]["target"]["width"] = 0.02
    elif change == "state":
        data["states"][0]["state"][0] += 0.01
    else:
        data["source"]["gravity_m_s2"] = 9.81
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        compare(runs, tmp_path / "comparison")
