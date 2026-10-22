"""Paper dynamics against independent mechanics and actual MuJoCo integration."""

import mujoco
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from exobalancelab.models import compile_model, human_xml, paper_xml
from exobalancelab.paper import PROFILES, exo_torque, ground_reaction, limit_human
from exobalancelab.simulation import RunConfig, run


@pytest.mark.parametrize("profile", PROFILES.values(), ids=PROFILES.keys())
def test_mujoco_acceleration_matches_paper_equation(profile):
    model, data = compile_model(paper_xml(profile))
    rng = np.random.default_rng(240805418)
    for _ in range(100):
        q, velocity = rng.uniform(-0.3, 0.3), rng.uniform(-1, 1)
        human = rng.uniform(profile.torque_min_nm, profile.torque_max_nm)
        exo = rng.uniform(-50, 50)
        data.qpos[0], data.qvel[0], data.ctrl[:] = q, velocity, [human, exo]
        mujoco.mj_forward(model, data)
        # Independent Eq. (3), expressed in the paper's absolute angle.
        theta = np.pi / 2 + q
        expected = (
            -9.81 / profile.length_m * np.cos(theta)
            + (human + exo - profile.damping_nm_s * velocity) / profile.inertia_kg_m2
        )
        assert data.qacc[0] == pytest.approx(expected, abs=2e-8)


def test_paper_integrator_converges_to_independent_ode():
    p = PROFILES["OF"]

    def derivative(t, state):
        angle, speed = state
        torque = -10
        return [
            speed,
            9.81 / p.length_m * np.sin(angle) + (torque - p.damping_nm_s * speed) / p.inertia_kg_m2,
        ]

    ref = solve_ivp(derivative, [0, 0.5], [-0.03, 0.2], rtol=1e-11, atol=1e-13).y[:, -1]
    errors = []
    for dt in [0.002, 0.001, 0.0005]:
        model, data = compile_model(paper_xml(p, dt))
        data.qpos[0], data.qvel[0], data.ctrl[0] = -0.03, 0.2, -10
        for _ in range(round(0.5 / dt)):
            mujoco.mj_step(model, data)
        errors.append(np.linalg.norm(np.r_[data.qpos, data.qvel] - ref))
    assert errors[-1] < 0.0005
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1]


def test_gravity_compensation_and_cop_static_equilibrium():
    p = PROFILES["OF"]
    q = 0.03
    total = -p.stiffness_nm * np.sin(q)
    grf = ground_reaction(q, 0, total, p)
    foot_mass = p.mass_kg - p.body_mass_kg
    com_x = (p.body_mass_kg * p.length_m * np.sin(q) + foot_mass * 0.31 * p.foot_length_m) / p.mass_kg
    assert grf["cop_m"] == pytest.approx(com_x)
    assert grf["normal_n"] == pytest.approx(p.mass_kg * 9.81)
    assert grf["friction_n"] == pytest.approx(0, abs=1e-12)
    assert exo_torque(q, 0, p, "gc") == pytest.approx(total)
    assert exo_torque(0.3, 0.1, p, "pd") == -50
    assert exo_torque(-0.3, -0.1, p, "pd") == 50


def test_human_rate_limits_track_applied_torque_after_saturation():
    p = PROFILES["OF"]
    previous = p.torque_max_nm
    assert limit_human(1000, previous, 0.001, p) == previous
    assert limit_human(-1000, previous, 0.001, p) == pytest.approx(previous + p.rate_min_nm_s * 0.001)


def test_humanoid_mass_joints_and_separate_device_actuation():
    p = PROFILES["OF"]
    model, data = compile_model(human_xml(p, device_mass_kg=2))
    assert sum(model.body_mass) == pytest.approx(62)
    assert model.nq == model.nv == 9
    assert model.nu == 8 and model.neq == 0
    for i, human in [(6, 2), (7, 5)]:
        assert model.actuator_trnid[i, 0] == model.actuator_trnid[human, 0]
    model2, _ = compile_model(human_xml(p, device_mass_kg=0))
    assert sum(model2.body_mass) == pytest.approx(60)


def test_humanoid_stance_has_real_support_contact():
    episode = run(RunConfig(model="human", duration_s=0.3, lean_rad=0, velocity_rad_s=0))
    assert np.min(episode.arrays["normal_n"]) > 500
    assert episode.metrics["recovered"]
    assert episode.metrics["contact_valid_fraction"] == 1


def test_episode_does_not_overwrite_evidence(tmp_path):
    episode = run(RunConfig(duration_s=0.01))
    path = tmp_path / "episode"
    episode.save(path)
    with pytest.raises(FileExistsError):
        episode.save(path)


@pytest.mark.parametrize(
    "kwargs",
    [{"timestep_s": 0}, {"duration_s": float("nan")}, {"control_period_s": 0.0015}, {"model": "unknown"}],
)
def test_invalid_run_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        run(RunConfig(**kwargs))


def test_forward_lean_and_pressure_point_toward_long_toe_region():
    p = PROFILES["OF"]
    model, data = compile_model(paper_xml(p))
    data.qpos[0] = 0.1
    data.ctrl[0] = -p.stiffness_nm * np.sin(0.1)
    mujoco.mj_forward(model, data)
    assert data.subtree_com[model.body("human").id, 0] > 0
    assert ground_reaction(0.1, 0, data.ctrl[0], p)["cop_m"] > 0
    assert data.qacc[0] == pytest.approx(0, abs=1e-10)
    assert model.geom_pos[model.geom("foot_1").id, 0] > 0


def test_articulated_ankle_angle_matches_paper_lean_when_feet_flat():
    p = PROFILES["OF"]
    model, data = compile_model(human_xml(p))
    q = 0.1
    data.qpos[2] = q
    for side in ["left", "right"]:
        data.qpos[model.jnt_qposadr[model.joint(side + "_ankle").id]] = q
    mujoco.mj_forward(model, data)
    for side in ["left", "right"]:
        rotation = data.xmat[model.body(side + "_foot").id].reshape(3, 3)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
