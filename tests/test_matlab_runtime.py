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
        np.testing.assert_allclose(feedback.arrays["qpos"], baseline.arrays["qpos"], atol=1e-11, rtol=0)
        np.testing.assert_allclose(
            feedback.arrays["human_raw_nm"], baseline.arrays["human_raw_nm"], atol=1e-9, rtol=0
        )
        from exobalancelab.paper import PROFILES
        from exobalancelab.reachability import HJBPolicy, policy_contract

        # Real MATLAB v7.3 export; this linear value is an interface fixture.
        controller.engine.eval(
            "g.vs={[1.2;1.9],[-2;2],[-352;88]}; "
            "V=repmat(reshape([-88;22],[1,1,2,1]),[2,2,1,2]); "
            "d=struct('scaling',true,'vscale',2,'tauscale',4,'omega',2);",
            nargout=0,
        )
        policy_path = tmp_path / "hjb_export.mat"
        controller.engine.feval(
            "export_hjb_policy",
            str(policy_path),
            controller.engine.workspace["g"],
            controller.engine.workspace["V"],
            matlab.double([0, 0.5]),
            controller.engine.workspace["d"],
            policy_contract(PROFILES["OF"], "none"),
            nargout=0,
        )
        policy = HJBPolicy(policy_path)
        policy.validate_plant(PROFILES["OF"], "none")
        assert policy.horizon_s == 1
        assert policy.rate(1, 0, 1, 22, PROFILES["OF"]) == PROFILES["OF"].rate_min_nm_s
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


@pytest.mark.skipif(os.environ.get("EXOBALANCE_MATLAB_TEST") != "1", reason="Licensed MATLAB test is opt-in")
def test_nominal_matlab_contact_constraints_match_mujoco():
    import mujoco

    import matlab
    from exobalancelab.models import compile_model, paper_xml
    from exobalancelab.paper import PROFILES

    directory = Path(__file__).resolve().parents[1] / "matlab"
    rng = np.random.default_rng(240805418)
    with MatlabController("standing_pd", directory) as controller:
        engine = controller.engine
        for p in PROFILES.values():
            model, data = compile_model(paper_xml(p))
            q = rng.uniform(-0.5, 0.5, 100)
            speed = rng.uniform(-2, 2, 100)
            torque = rng.uniform(p.torque_min_nm, p.torque_max_nm, 100)
            states = np.column_stack([q + np.pi / 2, speed, torque])
            engine.workspace["X"] = matlab.double(states.tolist())
            engine.eval(
                f"p=struct('m',{p.body_mass_kg},'mf',{p.mass_kg - p.body_mass_kg},"
                f"'l',{p.length_m},'lf',{p.foot_length_m},'g',9.81,'bFric',{p.damping_nm_s},"
                f"'ank',{0.19 * p.foot_length_m},'c',{0.31 * p.foot_length_m},'b',{0.039 * p.height_m},"
                f"'torqueMin',{p.torque_min_nm},'torqueMax',{p.torque_max_nm},"
                "'Kp',0,'Kd',0,'muGrav',0,'scaling',false,'vscale',2,'tauscale',4);"
                "[margin,reaction]=nominal_contact_constraints({X(:,1),X(:,2),X(:,3)},p);"
                "p.scaling=true;"
                "[scaled_margin,scaled_reaction]=nominal_contact_constraints({X(:,1),2*X(:,2),4*X(:,3)},p);",
                nargout=0,
            )
            actual = engine.workspace["reaction"]
            scaled = engine.workspace["scaled_reaction"]
            expected = []
            for angle, velocity, human in zip(q, speed, torque, strict=True):
                data.qpos[0], data.qvel[0], data.ctrl[0] = angle, velocity, human
                mujoco.mj_forward(model, data)
                acceleration = data.qacc[0]
                fx = (
                    p.body_mass_kg * p.length_m * (np.cos(angle) * acceleration - np.sin(angle) * velocity**2)
                )
                fz = p.mass_kg * 9.81 - p.body_mass_kg * p.length_m * (
                    np.sin(angle) * acceleration + np.cos(angle) * velocity**2
                )
                moment = (
                    -human
                    + p.damping_nm_s * velocity
                    + (p.mass_kg - p.body_mass_kg) * 9.81 * 0.31 * p.foot_length_m
                    - 0.039 * p.height_m * fx
                )
                expected.append([fx, fz, moment])
            for i, name in enumerate(["friction_n", "normal_n", "moment_nm"]):
                np.testing.assert_allclose(
                    np.asarray(actual[name]).ravel(), np.array(expected)[:, i], atol=1e-6, rtol=0
                )
                np.testing.assert_allclose(actual[name], scaled[name], atol=1e-12, rtol=0)
            fx, fz, moment = np.array(expected).T
            admissible = (
                (fz > 0)
                & (abs(fx) < fz)
                & (moment > -0.19 * p.foot_length_m * fz)
                & (moment < 0.81 * p.foot_length_m * fz)
            )
            np.testing.assert_array_equal(np.asarray(engine.workspace["margin"]).ravel() > 0, admissible)
