"""Backend comparisons must reject different problems and retain field errors."""

import json
import shutil

import h5py
import numpy as np
import pytest

from exobalancelab.hjb_solver_comparison import compare


def test_backend_comparison_reports_disagreement_and_rejects_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    reference = tmp_path / "reference"
    reference.mkdir()
    summary = {
        "profile": "synthetic",
        "exo": "none",
        "direction": "forward",
        "method": "none",
        "target": {"test": [1, 2]},
        "source_model": {"m": 1},
        "wall_s": 1,
    }
    (reference / "summary.json").write_text(json.dumps(summary))
    with h5py.File(reference / "policy.mat", "w") as file:
        for key, values in {
            "theta_rad": [1, 2],
            "velocity_rad_s": [-1, 0, 1],
            "human_torque_nm": [-1, 1],
            "time_to_go_s": [0, 1],
        }.items():
            file[key] = np.array(values)[None, :]
        file["value"] = -np.ones((2, 2, 3, 2))
        file["contract_json"] = np.array([ord(c) for c in '{"gravity":9.8}'], dtype=np.uint16)
    candidate = tmp_path / "candidate"
    shutil.copytree(reference, candidate)
    with h5py.File(candidate / "policy.mat", "a") as file:
        file["value"][1] = 1
    result = compare(reference, candidate, tmp_path / "comparison")
    assert result["final"]["max_absolute_error"] == 2
    assert result["final"]["sign_disagreement_nodes"] == 12
    assert result["final"]["negative_set_jaccard"] == 0
    assert result["area_rad2_per_s"] == {"reference": 1, "candidate": 0}
    assert result["relative_area_difference"] == -1
    assert result["policy_sha256"]["reference"] != result["policy_sha256"]["candidate"]
    with h5py.File(candidate / "policy.mat", "a") as file:
        file["value"][0] = 0
    with pytest.raises(ValueError, match="initial"):
        compare(reference, candidate, tmp_path / "wrong_initial")
    with h5py.File(candidate / "policy.mat", "a") as file:
        file["value"][0] = -1
        file["contract_json"][:] = np.array([ord(c) for c in '{"gravity":9.9}'], dtype=np.uint16)
    with pytest.raises(ValueError, match="contracts"):
        compare(reference, candidate, tmp_path / "wrong_contract")
