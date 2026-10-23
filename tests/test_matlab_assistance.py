"""Real MATLAB checks of reconciled assistance, nondimensionalization and targets."""

import os
import shutil
from pathlib import Path

import h5py
import mujoco
import numpy as np
import pytest

from exobalancelab.bridge import MatlabController
from exobalancelab.models import compile_model, paper_xml
from exobalancelab.paper import AUTHORS_PROFILES, exo_torque, ground_reaction


@pytest.mark.skipif(
    not (os.environ.get("EXOBALANCE_HJB_SOURCE") and os.environ.get("EXOBALANCE_HJB_DEPS")),
    reason="Licensed MATLAB and pinned HJB/SDP dependencies are opt-in",
)
@pytest.mark.parametrize("stretch", [[0, 0, 0], [0.15, 0.2, 0]])
def test_frozen_goal_refinement_keeps_physics_and_matches_independent_box_field(tmp_path, stretch):
    import json

    import matlab
    from exobalancelab.hjb_target_geometry import box_value

    directory = Path(__file__).resolve().parents[1] / "matlab"
    arguments = [
        "profile",
        "YF",
        "exo",
        "none",
        "target_definition",
        "source_entry",
        "resolution",
        matlab.double([[101], [101], [101]]),
        "prepare_only",
        True,
        "align_zero",
        True,
        "quiet",
        True,
    ]
    with MatlabController("standing_pd", directory) as controller:
        with pytest.raises(matlab.engine.MatlabExecutionError, match="frozen target requires"):
            controller.engine.feval(
                "solve_paper_hjb",
                os.environ["EXOBALANCE_HJB_SOURCE"],
                os.environ["EXOBALANCE_HJB_DEPS"],
                str(tmp_path / "invalid_legacy"),
                "frozen_target_file",
                str(tmp_path / "original/computed_target.mat"),
                nargout=0,
            )
        for name, extra in [
            ("original", []),
            (
                "refined",
                [
                    "align_upright",
                    True,
                    "target_field",
                    "width_normalized",
                    "frozen_target_file",
                    str(tmp_path / "original/computed_target.mat"),
                ],
            ),
        ]:
            controller.engine.feval(
                "solve_paper_hjb",
                os.environ["EXOBALANCE_HJB_SOURCE"],
                os.environ["EXOBALANCE_HJB_DEPS"],
                str(tmp_path / name),
                *arguments,
                *extra,
                "grid_stretch_si",
                matlab.double([[v] for v in stretch]) if name == "refined" else matlab.double([[0]] * 3),
                nargout=0,
            )
    before = json.loads((tmp_path / "original/summary.json").read_text())
    after = json.loads((tmp_path / "refined/summary.json").read_text())
    assert before["target"] == after["target"]
    assert before["effective_profile"] == after["effective_profile"]
    with h5py.File(tmp_path / "refined/problem.mat") as source:
        scale = np.asarray(source["state_scale"]).ravel()
        axes = [
            np.asarray(source[k]).ravel() * s
            for k, s in zip(["theta_rad", "velocity_rad_s", "human_torque_nm"], scale, strict=True)
        ]
        assert min(abs(axes[0] - np.pi / 2)) < 1e-14
        assert min(abs(axes[1])) < 1e-14
        assert min(abs(axes[2])) < 1e-14
        target = after["target"]
        fields = []
        for key, label in [("ankleEq", "Ank"), ("toeEq", "Toe")]:
            widths = (
                np.array(
                    [
                        target[f"theta{label}Range"],
                        target[f"thetaDot{label}Range"],
                        target[f"torque{label}Range"],
                    ]
                )
                * scale
            )
            fields.append(box_value(axes, np.array(target[key]) * scale, widths))
        expected = np.maximum(np.minimum(*fields), np.asarray(source["obstacle_value"]).T)
        np.testing.assert_allclose(np.asarray(source["initial_value"]).T, expected, atol=2e-14, rtol=0)
        if any(stretch):
            from exobalancelab.paper import acceleration

            computational = [np.asarray(source[k]).ravel() for k in ["solver_theta", "solver_velocity"]]
            jacobians = []
            for dimension, coordinate in enumerate(computational):
                mapping = after["grid_mapping"][dimension]
                expected_axis = (
                    mapping["center"]
                    + mapping["lambda"] * np.sinh(coordinate)
                    + mapping["correction"] * coordinate**2
                )
                np.testing.assert_allclose(axes[dimension], expected_axis, atol=2e-14, rtol=0)
                jacobians.append(
                    mapping["lambda"] * np.cosh(coordinate) + 2 * mapping["correction"] * coordinate
                )
                assert min(jacobians[-1]) > 0
            q = axes[0][:, None, None] - np.pi / 2
            velocity = axes[1][None, :, None] / scale[1]
            torque = axes[2][None, None, :] / scale[2]
            time_scale = float(np.asarray(source["time_scale_s"]).item())
            expected = acceleration(q, velocity, torque, AUTHORS_PROFILES["YF"]) * time_scale * scale[1]
            restored = np.asarray(source["drift_velocity"]).T * jacobians[1][None, :, None]
            np.testing.assert_allclose(restored, expected, atol=2e-14, rtol=0)


@pytest.mark.skipif(
    not (os.environ.get("EXOBALANCE_HJB_SOURCE") and os.environ.get("EXOBALANCE_HJB_DEPS")),
    reason="Licensed MATLAB and pinned HJB/SDP dependencies are opt-in",
)
def test_matlab_strength_axes_are_independent_and_source_entry_targets_explicit(tmp_path):
    import json

    from exobalancelab.paper import scale_strength

    directory = Path(__file__).resolve().parents[1] / "matlab"
    with MatlabController("standing_pd", directory) as controller:
        for index, (mt, mrtd) in enumerate([(1, 1), (0.5, 1), (1, 0.5)]):
            output = tmp_path / str(index)
            controller.engine.feval(
                "solve_paper_hjb",
                os.environ["EXOBALANCE_HJB_SOURCE"],
                os.environ["EXOBALANCE_HJB_DEPS"],
                str(output),
                "profile",
                "YF",
                "exo",
                "gc",
                "target_definition",
                "source_entry",
                "mt_scale",
                float(mt),
                "mrtd_scale",
                float(mrtd),
                "prepare_only",
                True,
                "duration_s",
                0.001,
                "output_step_s",
                0.001,
                "quiet",
                True,
                nargout=0,
            )
            summary = json.loads((output / "summary.json").read_text())
            profile = scale_strength(AUTHORS_PROFILES["YF"], mt, mrtd)
            for key in ["torque_min_nm", "torque_max_nm", "rate_min_nm_s", "rate_max_nm_s"]:
                assert summary["effective_profile"][key] == getattr(profile, key)
            assert summary["target"]["thetaDotAnkRange"] == 0.01
            assert summary["target"]["thetaDotToeRange"] == 0.01
            assert summary["target"]["torqueAnkRange"] == 1
            with h5py.File(output / "problem.mat") as source:
                scale = np.asarray(source["state_scale"]).ravel()
                time_scale = float(np.asarray(source["time_scale_s"]).item())
                actual = np.asarray(source["rate_bounds"]).ravel() / scale[2] / time_scale
                np.testing.assert_allclose(actual, [profile.rate_min_nm_s, profile.rate_max_nm_s], atol=1e-10)


@pytest.mark.skipif(
    not (os.environ.get("EXOBALANCE_HJB_SOURCE") and os.environ.get("EXOBALANCE_HJB_DEPS")),
    reason="Licensed MATLAB and pinned HJB dependencies are opt-in",
)
def test_prepared_hjb_fields_equal_the_native_solver_inputs(tmp_path):
    directory = Path(__file__).resolve().parents[1] / "matlab"
    with MatlabController("standing_pd", directory) as controller:
        for prepared, name in [(True, "prepared"), (False, "native")]:
            controller.engine.feval(
                "solve_paper_hjb",
                os.environ["EXOBALANCE_HJB_SOURCE"],
                os.environ["EXOBALANCE_HJB_DEPS"],
                str(tmp_path / name),
                "prepare_only",
                prepared,
                "duration_s",
                0.001,
                "output_step_s",
                0.001,
                "quiet",
                True,
                nargout=0,
            )
    with (
        h5py.File(tmp_path / "prepared/problem.mat") as prepared,
        h5py.File(tmp_path / "native/policy.mat") as native,
    ):
        for key in ["theta_rad", "velocity_rad_s", "human_torque_nm", "time_to_go_s"]:
            np.testing.assert_allclose(prepared[key], native[key], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(prepared["initial_value"], native["value"][0])
        assert np.count_nonzero(np.asarray(prepared["initial_value"]) < 0) > 0
        assert not (tmp_path / "prepared/policy.mat").exists()


@pytest.mark.skipif(
    not (os.environ.get("EXOBALANCE_HJB_SOURCE") and os.environ.get("EXOBALANCE_HJB_DEPS")),
    reason="Licensed MATLAB and pinned HJB/SDP dependencies are opt-in",
)
def test_reconciled_matlab_assistance_matches_mujoco_and_target_constraints(tmp_path):
    import matlab

    source = Path(os.environ["EXOBALANCE_HJB_SOURCE"])
    dependencies = Path(os.environ["EXOBALANCE_HJB_DEPS"])
    shutil.copytree(source / "Dynamics/LiftedInvPendExo", tmp_path / "@LiftedInvPendExo")
    directory = Path(__file__).resolve().parents[1] / "matlab"
    rng = np.random.default_rng(240805418)
    max_error = 0.0
    with MatlabController("standing_pd", directory) as controller:
        engine = controller.engine
        for name in ["helperOC", "YALMIP", "sedumi"]:
            engine.addpath(engine.genpath(str(dependencies / name)), nargout=0)
        for path in [tmp_path, source / "Models/ModelUtils"]:
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
            for mode in ["none", "gc", "pd"]:
                engine.workspace["mode"] = mode
                engine.eval(
                    "md=load(model_file); d=PaperExoPendulum(md.model,mode);"
                    "device=d.exoTorqueSI(X(:,1),X(:,2));"
                    "dx=d.dynamics(0,{X(:,1),X(:,2)*d.vscale,X(:,3)*d.tauscale},U*d.rtdscale,[]);"
                    "physical_dx=[dx{1}/d.vscale,dx{2}/(d.omega*d.vscale),dx{3}/d.rtdscale];"
                    "[margin,reaction]=paper_contact_constraints("
                    "{X(:,1),X(:,2)*d.vscale,X(:,3)*d.tauscale},d,device);"
                    "d.scaling=false; dx=d.dynamics(0,{X(:,1),X(:,2),X(:,3)},U,[]);"
                    "unscaled_dx=[dx{1},dx{2},dx{3}];"
                    "[margin_si,reaction_si]=paper_contact_constraints({X(:,1),X(:,2),X(:,3)},d,device);",
                    nargout=0,
                )
                actual = np.asarray(engine.workspace["physical_dx"])
                np.testing.assert_allclose(actual, np.asarray(engine.workspace["unscaled_dx"]), atol=1e-12)
                np.testing.assert_allclose(
                    engine.workspace["margin"], engine.workspace["margin_si"], atol=1e-12
                )
                device = np.asarray(engine.workspace["device"]).ravel()
                reaction = engine.workspace["reaction"]
                model, data = compile_model(paper_xml(p))
                for index, (theta, speed, torque) in enumerate(states):
                    q = theta - np.pi / 2
                    data.qpos[0], data.qvel[0] = q, speed
                    data.ctrl[:] = [torque, exo_torque(q, speed, p, mode)]
                    mujoco.mj_forward(model, data)
                    expected = [speed, data.qacc[0], rates[index]]
                    max_error = max(max_error, float(np.max(abs(actual[index] - expected))))
                    np.testing.assert_allclose(actual[index], expected, atol=2e-8, rtol=0)
                    assert device[index] == pytest.approx(data.ctrl[1], abs=1e-12)
                    mass, length = p.body_mass_kg, p.length_m
                    fx = mass * length * (np.cos(q) * data.qacc[0] - np.sin(q) * speed**2)
                    fz = p.mass_kg * p.gravity_m_s2 - mass * length * (
                        np.sin(q) * data.qacc[0] + np.cos(q) * speed**2
                    )
                    assert np.asarray(reaction["normal_n"]).ravel()[index] == pytest.approx(fz, abs=2e-7)
                    assert np.asarray(reaction["friction_n"]).ravel()[index] == pytest.approx(fx, abs=2e-7)
                if mode == "none":
                    continue
                engine.eval(
                    "d.scaling=true; [target,td]=paper_target_boxes(d);"
                    "target_check=[[td.linear_constraint_max_residual];"
                    "[td.minimum_corner_contact_margin]];",
                    nargout=0,
                )
                check = np.asarray(engine.workspace["target_check"])
                assert np.max(check[0]) < 1e-9
                assert np.min(check[1]) > 0
                target = engine.workspace["target"]
                for label, center_key in [("Ank", "ankleEq"), ("Toe", "toeEq")]:
                    center = np.asarray(target[center_key]).ravel()
                    widths = np.array(
                        [
                            target[f"theta{label}Range"],
                            target[f"thetaDot{label}Range"],
                            target[f"torque{label}Range"],
                        ],
                        dtype=float,
                    ).ravel()
                    assert np.all(widths > 0)
                    for signs in np.ndindex(2, 2, 2):
                        theta, speed, torque = center + (2 * np.array(signs) - 1) * widths
                        q = theta - np.pi / 2
                        total = torque + exo_torque(q, speed, p, mode)
                        grf = ground_reaction(q, speed, total, p)
                        assert min(grf["normal_n"], grf["cop_margin_m"], grf["friction_margin_n"]) > 0
                        assert p.torque_min_nm <= torque <= p.torque_max_nm
    print(f"Reconciled MATLAB/MuJoCo max derivative error: {max_error:.12g}")
