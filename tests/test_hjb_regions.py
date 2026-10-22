"""Synthetic rectangular regions test campaign completeness and area accounting."""

import json

import numpy as np
import pytest
from scipy.io import savemat

from exobalancelab.hjb_regions import DIRECTIONS, MODES, PROFILES, compare_grids, report
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


def test_grid_report_separates_indicator_quadrature_from_identical_value_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    profile = AUTHORS_PROFILES["OF"]
    sizes = [3, 5, 9]
    cases = []
    for size in sizes:
        for mode in MODES:
            for direction in DIRECTIONS:
                folder = tmp_path / f"{mode}_{direction}_{size}"
                folder.mkdir()
                case = {
                    "profile": "OF",
                    "exo": mode,
                    "direction": direction,
                    "requested_resolution": [size] * 3,
                    "solved": str(folder),
                }
                cases.append(case)
                summary = dict(
                    case,
                    stage="solved",
                    requested_resolution=[size] * 3,
                    resolution=[size] * 3,
                    duration_s=1.4,
                    output_step_s=0.01,
                    velocity_buffer_rad_s=0.5,
                    method="none",
                    aligned_zero=True,
                    wall_s=1,
                    solver_sha256="synthetic-solver",
                    precision="float64",
                    target={"thetaAnkRange": 0.01},
                    gravity_m_s2=profile.gravity_m_s2,
                    source_model={
                        "m": profile.mass_kg,
                        "h": profile.height_m,
                        "tauMin": profile.torque_min_nm,
                        "tauMax": profile.torque_max_nm,
                        "uMin": profile.rate_min_nm_s,
                        "uMax": profile.rate_max_nm_s,
                    },
                )
                (folder / "summary.json").write_text(json.dumps(summary))
                theta, velocity = np.linspace(0, 2, size), np.linspace(-1, 1, size)
                # The same exact linear value function on every grid: V = theta - 1.
                savemat(
                    folder / "projection.mat",
                    {
                        "theta_rad": theta,
                        "velocity_rad_s": velocity,
                        "projected_value": np.broadcast_to((theta - 1)[:, None], (size, size)),
                    },
                )
    protocol = {
        "parameter_set": "authors",
        "target_sets": "Synthetic linear field",
        "cases": cases,
        "requested_resolutions": [[size] * 3 for size in sizes],
        "horizon_s": 1.4,
        "output_step_s": 0.01,
        "velocity_buffer_rad_s": 0.5,
        "method": "none",
        "align_zero": True,
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    actual = compare_grids(path, tmp_path / "report")
    assert len(actual["areas"]) == 18 and len(actual["comparisons"]) == 12
    for row in actual["areas"]:
        size = int(row["requested_resolution"].split("x")[0])
        assert row["area_rad2_per_s"] == 1 + 1 / (size - 1)
    for row in actual["comparisons"]:
        assert row["native_grid_area_change_percent"] < 0
        assert row["common_negative_set_jaccard"] == 1
        assert row["common_sign_disagreement_nodes"] == 0
        assert row["common_coarse_area_rad2_per_s"] == row["common_fine_area_rad2_per_s"]
    # Extending the fine velocity domain must not extrapolate the coarse field.
    theta, velocity = np.linspace(0, 2, 9), np.linspace(-1.5, 1.5, 9)
    savemat(
        tmp_path / "pd_backward_9/projection.mat",
        {
            "theta_rad": theta,
            "velocity_rad_s": velocity,
            "projected_value": np.broadcast_to((theta - 1)[:, None], (9, 9)),
        },
    )
    extended = compare_grids(path, tmp_path / "extended")
    paired = next(
        r
        for r in extended["comparisons"]
        if r["exo"] == "pd" and r["direction"] == "backward" and r["fine_requested_resolution"] == "9x9x9"
    )
    assert paired["common_velocity_min_rad_s"] == -0.75
    assert paired["common_velocity_max_rad_s"] == 0
    assert paired["common_negative_set_jaccard"] == 1
    changed = tmp_path / "pd_backward_9/summary.json"
    original = json.loads(changed.read_text())
    changed.write_text(json.dumps(dict(original, target={"thetaAnkRange": 0.02})))
    with pytest.raises(ValueError, match="Different target"):
        compare_grids(path, tmp_path / "changed_target")
    changed.write_text(json.dumps(dict(original, solver_sha256="different")))
    with pytest.raises(ValueError, match="Different solver"):
        compare_grids(path, tmp_path / "changed_solver")
    changed.write_text(json.dumps(original))
    # Dropping a whole declared level must not silently produce a partial study.
    protocol["cases"] = [c for c in cases if c["requested_resolution"][0] != 9]
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="every resolution"):
        compare_grids(path, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()
