"""Opt-in cross-check of unchanged authors' MATLAB dynamics against MuJoCo.

Set EXOBALANCE_HJB_SOURCE and EXOBALANCE_HJB_DEPS to the pinned source directories.
"""

import os
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest

from exobalancelab.bridge import MatlabController
from exobalancelab.models import compile_model, paper_xml
from exobalancelab.paper import AUTHORS_PROFILES


@pytest.mark.skipif(
    not (os.environ.get("EXOBALANCE_HJB_SOURCE") and os.environ.get("EXOBALANCE_HJB_DEPS")),
    reason="Licensed MATLAB and pinned authors/helperOC sources are opt-in",
)
def test_original_matlab_lifted_dynamics_match_native_mujoco(tmp_path):
    import matlab

    source = Path(os.environ["EXOBALANCE_HJB_SOURCE"])
    dependencies = Path(os.environ["EXOBALANCE_HJB_DEPS"])
    shutil.copytree(source / "Dynamics/LiftedInvPendExo", tmp_path / "@LiftedInvPendExo")
    directory = Path(__file__).resolve().parents[1] / "matlab"
    rng = np.random.default_rng(240805418)
    max_error = 0.0
    with MatlabController("standing_pd", directory) as controller:
        engine = controller.engine
        engine.addpath(engine.genpath(str(dependencies / "helperOC")), nargout=0)
        for path in [tmp_path, source / "Models/ModelUtils", source / "Dynamics/DynUtils"]:
            engine.addpath(str(path), nargout=0)
        for p in AUTHORS_PROFILES.values():
            states = np.column_stack(
                [
                    rng.uniform(-0.5, 0.5, 100) + np.pi / 2,
                    rng.uniform(-2, 2, 100),
                    rng.uniform(p.torque_min_nm, p.torque_max_nm, 100),
                ]
            )
            rates = rng.uniform(p.rate_min_nm_s, p.rate_max_nm_s, 100)
            engine.workspace["X"] = matlab.double(states.tolist())
            engine.workspace["U"] = matlab.double(rates[:, None].tolist())
            engine.workspace["model_file"] = str(source / "Models" / (p.name + "_model.mat"))
            engine.workspace["target_file"] = str(
                source / "TargetSets/computedSets" / ("Params_" + p.name + "_noExo_100_pctMT_100_pctRTD.mat")
            )
            engine.eval(
                "md=load(model_file); td=load(target_file);"
                "d=LiftedInvPendExo([pi/2;0;0],md.model,td.problemParams);"
                "dx=d.dynamics(0,{X(:,1),X(:,2)*d.vscale,X(:,3)*d.tauscale},U*d.rtdscale,[]);"
                "physical_dx=[dx{1}/d.vscale,dx{2}/(d.omega*d.vscale),dx{3}/d.rtdscale];",
                nargout=0,
            )
            actual = np.asarray(engine.workspace["physical_dx"])
            model, data = compile_model(paper_xml(p))
            for i, (theta, velocity, torque) in enumerate(states):
                data.qpos[0], data.qvel[0], data.ctrl[0] = theta - np.pi / 2, velocity, torque
                mujoco.mj_forward(model, data)
                expected = [velocity, data.qacc[0], rates[i]]
                max_error = max(max_error, float(np.max(abs(actual[i] - expected))))
                np.testing.assert_allclose(actual[i], expected, atol=2e-8, rtol=0)
    print(f"Original MATLAB / MuJoCo maximum derivative error: {max_error:.12g}")
