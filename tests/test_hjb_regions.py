"""Synthetic rectangular regions test campaign completeness and area accounting."""

import json

import numpy as np
import pytest
from scipy.io import savemat

from exobalancelab.hjb_regions import DIRECTIONS, MODES, PROFILES, report
from exobalancelab.paper import AUTHORS_PROFILES


def test_campaign_report_integrates_both_directions_and_rejects_missing_or_changed_models(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    cases = []
    for profile in PROFILES:
        p = AUTHORS_PROFILES[profile]
        for mode in MODES:
            for direction in DIRECTIONS:
                folder = tmp_path / f"{profile}_{mode}_{direction}"
                folder.mkdir()
                cases.append({"profile": profile, "exo": mode, "direction": direction, "solved": str(folder)})
                metadata = {
                    "profile": profile,
                    "exo": mode,
                    "direction": direction,
                    "stage": "solved",
                    "method": "none",
                    "aligned_zero": True,
                    "duration_s": 1.4,
                    "output_step_s": 0.01,
                    "velocity_buffer_rad_s": 0.5,
                    "resolution": [3, 3, 3],
                    "requested_resolution": [3, 3, 3],
                    "wall_s": 1,
                    "gravity_m_s2": p.gravity_m_s2,
                    "source_model": {
                        "m": p.mass_kg,
                        "h": p.height_m,
                        "tauMin": p.torque_min_nm,
                        "tauMax": p.torque_max_nm,
                        "uMin": p.rate_min_nm_s,
                        "uMax": p.rate_max_nm_s,
                    },
                }
                (folder / "summary.json").write_text(json.dumps(metadata))
                negative_theta = {
                    "none": [True, True, True],
                    "gc": [True, True, False],
                    "pd": [False, True, False],
                }[mode]
                values = np.broadcast_to(np.where(negative_theta, -1, 1)[:, None], (3, 3))
                savemat(
                    folder / "projection.mat",
                    {
                        "theta_rad": [0, 1, 2],
                        "velocity_rad_s": [-1, 0, 1],
                        "projected_value": values,
                    },
                )
    protocol = {
        "parameter_set": "authors",
        "target_sets": "Synthetic test fixture, not HJB evidence",
        "cases": cases,
        "horizon_s": 1.4,
        "output_step_s": 0.01,
        "velocity_buffer_rad_s": 0.5,
        "requested_resolution": [3, 3, 3],
        "method": "none",
        "align_zero": True,
        "paper_table_ii_percent": {
            p: {f"{m}_{d}": 0 for m in ["gc", "pd"] for d in DIRECTIONS} for p in PROFILES
        },
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    actual = report(path, tmp_path / "report")
    assert len(actual["areas"]) == 30 and len(actual["table_ii"]) == 20
    for row in actual["areas"]:
        assert row["area_rad2_per_s"] == {"none": 2, "gc": 1.5, "pd": 1}[row["exo"]]
    for row in actual["table_ii"]:
        assert row["observed_change_percent"] == {"gc": -25, "pd": -50}[row["exo"]]
    model_path = tmp_path / "YM_none_forward/summary.json"
    original = json.loads(model_path.read_text())
    model_path.write_text(json.dumps(dict(original, gravity_m_s2=9.81)))
    with pytest.raises(ValueError, match="physical model"):
        report(path, tmp_path / "different_model")
    model_path.write_text(json.dumps(dict(original, stage="running")))
    with pytest.raises(ValueError, match="Incomplete HJB campaign"):
        report(path, tmp_path / "unfinished")
    assert not (tmp_path / "unfinished").exists()
