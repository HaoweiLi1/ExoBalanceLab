"""Opt-in licensed MATLAB integration, separate from synthetic transport tests.

Run EXOBALANCE_MATLAB_TEST=1 pytest -q -s tests/test_matlab_runtime.py after activation.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from exobalancelab.bridge import MatlabController, Trajectory
from exobalancelab.simulation import RunConfig, run


@pytest.mark.skipif(os.environ.get("EXOBALANCE_MATLAB_TEST") != "1", reason="Licensed MATLAB test is opt-in")
def test_real_matlab_export_and_closed_loop_match_python(tmp_path):
    import matlab

    directory = Path(__file__).resolve().parents[1] / "matlab"
    config = RunConfig(duration_s=0.2, control_period_s=0.01, lean_rad=0.01, velocity_rad_s=0)
    baseline = run(config)
    with MatlabController("standing_pd", directory) as controller:
        time = np.linspace(0, 0.2, 21)
        torque = np.sin(time)
        for extension in ["csv", "mat"]:
            path = tmp_path / ("matlab_export." + extension)
            controller.engine.feval(
                "export_controller",
                str(path),
                matlab.double(time[:, None].tolist()),
                matlab.double(torque[:, None].tolist()),
                nargout=0,
            )
            loaded = Trajectory.load(path)
            np.testing.assert_allclose(loaded.values[:, 0], torque, atol=1e-14)
        feedback = run(config, matlab=controller)
        np.testing.assert_allclose(feedback.arrays["qpos"], baseline.arrays["qpos"], atol=1e-11)
        np.testing.assert_allclose(
            feedback.arrays["human_raw_nm"], baseline.arrays["human_raw_nm"], atol=1e-9
        )
        print(
            json.dumps(
                {
                    "matlab_release": controller.engine.version("-release"),
                    "calls": len(controller.latency_s),
                    "engine_p50_ms": float(np.median(controller.latency_s) * 1000),
                    "engine_p99_ms": float(np.percentile(controller.latency_s, 99) * 1000),
                }
            )
        )
