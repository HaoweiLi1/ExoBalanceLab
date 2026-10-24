"""Deterministic MuJoCo episodes: control at t, zero-order hold, then physics step.

Records pair states and applied commands at the SAME pre-step time. All quantities
are SI; paper ankle q = theta-pi/2. Rendering is an offline operation, never a
control-loop dependency. Outputs retain raw/applied actions and the model XML.
"""

import hashlib
import json
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from exobalancelab.bridge import MatlabController, Trajectory
from exobalancelab.models import compile_model, human_xml, paper_xml
from exobalancelab.paper import (
    PROFILE_SETS,
    PROFILES,
    exo_torque,
    ground_reaction,
    limit_human,
    scale_strength,
)
from exobalancelab.reachability import TIME_TOLERANCE_S, HJBPolicy


@dataclass
class RunConfig:
    """Episode configuration; timestep and controller period are separate seconds."""

    model: str = "paper"
    profile: str = "OF"
    parameters: str = "table_i"
    exo: str = "none"
    duration_s: float = 6.0
    timestep_s: float = 0.001
    control_period_s: float = 0.001
    lean_rad: float = -0.03
    velocity_rad_s: float = 0.15
    initial_torque_nm: float = 0.0
    device_mass_kg: float = 2.0
    push_n: float = 0.0
    push_start_s: float = 1.0
    push_duration_s: float = 0.15
    human_strength: float = 1.0
    mt_scale: float = 1.0
    mrtd_scale: float = 1.0
    command_delay_s: float = 0.0
    human_feedback: str = "default"
    initial_posture: str = "root"

    def validate(self) -> None:
        if self.model not in {"paper", "contact", "human"} or self.profile not in PROFILES:
            raise ValueError("Unknown model or Table I profile")
        if self.parameters not in PROFILE_SETS:
            raise ValueError("Parameters must be table_i or authors")
        if self.exo not in {"none", "gc", "pd"}:
            raise ValueError("Exoskeleton must be none, gc or pd")
        if self.human_feedback not in {"default", "fixed", "readapted"}:
            raise ValueError("Human feedback must be default, fixed or readapted")
        if self.initial_posture not in {"root", "flat_feet"}:
            raise ValueError("Initial posture must be root or flat_feet")
        if self.initial_posture == "flat_feet" and self.model != "human":
            raise ValueError("Flat-feet initialization is for the articulated model")
        values = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.isfinite(values).all() or min(self.duration_s, self.timestep_s, self.control_period_s) <= 0:
            raise ValueError("Configuration numbers must be finite and times positive")
        for period in [self.duration_s, self.control_period_s, self.command_delay_s]:
            if not np.isclose(period / self.timestep_s, round(period / self.timestep_s), atol=1e-9, rtol=0):
                raise ValueError("Duration, control period and delay must be integer physics-step multiples")
        if self.control_period_s < self.timestep_s or not 0 < self.human_strength <= 1:
            raise ValueError("Control period must cover a physics step; human_strength must be in (0,1]")
        p = PROFILE_SETS[self.parameters][self.profile]
        p = scale_strength(p, self.mt_scale, self.mrtd_scale)
        if not (
            p.torque_min_nm * self.human_strength
            <= self.initial_torque_nm
            <= p.torque_max_nm * self.human_strength
        ):
            raise ValueError("Initial human torque exceeds strength-scaled Table I limits")
        if min(self.device_mass_kg, self.push_duration_s, self.push_start_s, self.command_delay_s) < 0:
            raise ValueError("Device mass and push timing must be nonnegative")
        if self.model != "human" and self.push_n != 0:
            raise ValueError("External chest pushes are supported by the articulated human model")


@dataclass
class Episode:
    """Dense pre-step observations, summary and self-contained model/configuration."""

    config: RunConfig
    xml: str
    arrays: dict[str, np.ndarray]
    metrics: dict
    controller_samples: dict[str, np.ndarray] | None = None

    def save(self, directory: str | Path) -> None:
        """Write a new directory; never overwrite an existing experiment."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        (path / "model.xml").write_text(self.xml)
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        (path / "metrics.json").write_text(json.dumps(self.metrics, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(path / "episode.npz", **self.arrays)
        if self.controller_samples is not None:
            np.savez_compressed(path / "controller_samples.npz", **self.controller_samples)
        columns = [name for name, value in self.arrays.items() if value.ndim == 1]
        np.savetxt(
            path / "signals.csv",
            np.column_stack([self.arrays[name] for name in columns]),
            delimiter=",",
            header=",".join(columns),
            comments="",
        )
        provenance = {"mujoco": mujoco.__version__, "numpy": np.__version__}
        provenance["source_sha256"] = {
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()
            for source in Path(__file__).parent.glob("*.py")
        }
        provenance["model_sha256"] = hashlib.sha256(self.xml.encode()).hexdigest()
        provenance["git_commit"] = None
        provenance["working_tree_dirty"] = None
        source_root = Path(__file__).resolve().parents[1]
        # A wheel has no checkout; never inherit an unrelated parent/caller repository.
        if (source_root / ".git").exists():
            try:
                git = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True
                )
                status = subprocess.run(
                    ["git", "status", "--porcelain"], cwd=source_root, capture_output=True, text=True
                )
            except FileNotFoundError:
                pass  # Git is optional; source and model hashes still identify the run.
            else:
                provenance["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
                provenance["working_tree_dirty"] = (
                    bool(status.stdout.strip()) if status.returncode == 0 else None
                )
        (path / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def _human_feedback(model, data, profile, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Settle vertical contact then linearize the actual MuJoCo plant for stance LQR.

    This is a synthetic full-state human controller. The device is not optimized.
    Fixed feedback uses the unassisted plant. Readaptation includes the sampled
    device feedback in the linearization, with an unchanged reference pose.
    Settling and controller design use a fixed <=1 ms step independent of the
    rollout step, so a physics refinement cannot silently retune the controller.
    """
    physics_step = model.opt.timestep
    design_steps = max(1, int(np.ceil(config.control_period_s / 0.001)))
    model.opt.timestep = config.control_period_s / design_steps
    # Symmetric vertical settling establishes supporting contacts without pose teleporting.
    for _ in range(round(0.7 / model.opt.timestep)):
        data.ctrl[:] = 0
        for i in range(6):
            joint = model.actuator_trnid[i, 0]
            qidx, vidx = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
            data.ctrl[i] = -600 * data.qpos[qidx] - 40 * data.qvel[vidx]
        mujoco.mj_step(model, data)
    reference = data.qpos.copy()
    data.qvel[:] = 0
    data.ctrl[:] = 0
    data.time = 0
    mujoco.mj_forward(model, data)
    n = 2 * model.nv
    a, b = np.empty((n, n)), np.empty((n, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, a, b, None, None)
    # Lift the physics transition to the actual zero-order-hold controller period.
    ac, bc = np.eye(n), np.zeros_like(b)
    for _ in range(design_steps):
        bc = a @ bc + b
        ac = a @ ac
    feedforward = np.zeros(6)
    if config.human_feedback == "readapted":
        derivative = np.zeros((2, n))
        for i, side in enumerate(["left", "right"]):
            joint = model.joint(side + "_ankle").id
            qi, vi = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
            q = reference[qi]
            feedforward[2 + 3 * i] = -0.5 * exo_torque(q, 0, profile, config.exo)
            for axis, dq, dv in [(qi, 1e-6, 0), (model.nv + vi, 0, 1e-6)]:
                derivative[i, axis] = (
                    0.5
                    * (
                        exo_torque(q + dq, dv, profile, config.exo)
                        - exo_torque(q - dq, -dv, profile, config.exo)
                    )
                    / (2e-6)
                )
        ac += bc[:, 6:] @ derivative
    weights = np.r_[np.array([1200, 1200, 1600] + [180] * 6), np.array([80, 80, 100] + [12] * 6)]
    r = np.eye(6) * 0.015
    p = solve_discrete_are(ac, bc[:, :6], np.diag(weights), r)
    gain = np.linalg.solve(r + bc[:, :6].T @ p @ bc[:, :6], bc[:, :6].T @ p @ ac)
    model.opt.timestep = physics_step
    mujoco.mj_forward(model, data)
    return reference, gain, feedforward


def _contact_signals(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float, float]:
    """Measured floor contact normal/friction sums and load-weighted world CoP."""
    force = np.zeros(6)
    total = np.zeros(3)
    weighted_x = 0.0
    floor = model.geom("ground").id
    for i in range(data.ncon):
        contact = data.contact[i]
        if floor not in (contact.geom1, contact.geom2):
            continue
        mujoco.mj_contactForce(model, data, i, force)
        world = contact.frame.reshape(3, 3).T @ force[:3]
        if contact.geom2 == floor:
            world = -world
        total += world
        weighted_x += contact.pos[0] * world[2]
    return float(total[2]), float(total[0]), float(weighted_x / total[2]) if total[2] > 1e-6 else np.nan


def run(
    config: RunConfig,
    trajectory: Trajectory | None = None,
    matlab: MatlabController | None = None,
    hjb: HJBPolicy | None = None,
) -> Episode:
    """Run an episode with bounded human feedback, offline replay or MATLAB feedback.

    Replay and MATLAB outputs specify desired HUMAN ankle torque, not total torque.
    For the human model they require six channels: L hip/knee/ankle, R hip/knee/ankle.
    Human C1/C2 limits are applied after the external controller and logged separately.
    HJB instead specifies a held torque rate, integrated once per physics step.
    """
    config.validate()
    if sum(item is not None for item in [trajectory, matlab, hjb]) > 1:
        raise ValueError("Choose one external human controller")
    if hjb is not None:
        if (
            config.model != "paper"
            or config.duration_s - hjb.horizon_s > TIME_TOLERANCE_S
            or hjb.minimum_horizon_s > TIME_TOLERANCE_S
            or getattr(hjb, "minimum_duration_s", 0) - config.duration_s > TIME_TOLERANCE_S
        ):
            raise ValueError(
                "HJB requires the paper model, a supported duration and complete [0,duration] time coverage"
            )
    p = PROFILE_SETS[config.parameters][config.profile]
    p = scale_strength(p, config.human_strength * config.mt_scale, config.human_strength * config.mrtd_scale)
    if hjb is not None:
        hjb.validate_plant(p, config.exo)
    xml = (
        human_xml(p, config.timestep_s, config.device_mass_kg)
        if config.model == "human"
        else paper_xml(p, config.timestep_s, contact=config.model == "contact")
    )
    model, data = compile_model(xml)
    human = config.model == "human"
    channels = 6 if human else 1
    if trajectory is not None:
        if trajectory.values.shape[1] != channels:
            raise ValueError(f"Expected {channels} human torque channels")
        trajectory.sample(0)
        trajectory.sample(config.duration_s - config.timestep_s)
    if human:
        if config.initial_torque_nm != 0:
            raise ValueError("Scalar initial torque is supported only by the paper/contact model")
        reference, gain, feedforward = _human_feedback(model, data, p, config)
        feet = [model.geom(side + "_sole").id for side in ["left", "right"]]
        settled_feet = data.geom_xpos[feet].mean(axis=0).copy()
        data.qpos[2] += config.lean_rad
        data.qvel[2] = config.velocity_rad_s
        if config.initial_posture == "flat_feet":
            # Rotate straight legs about the supporting feet, keeping sole pose
            # and velocity unchanged. A root-only rotation tilts the feet at t=0.
            for side in ["left", "right"]:
                joint = model.joint(side + "_ankle").id
                data.qpos[model.jnt_qposadr[joint]] += config.lean_rad
                data.qvel[model.jnt_dofadr[joint]] = config.velocity_rad_s
            mujoco.mj_forward(model, data)
            displacement = settled_feet - data.geom_xpos[feet].mean(axis=0)
            data.qpos[0] += displacement[0]
            data.qpos[1] += displacement[2]
            mujoco.mj_forward(model, data)
            jacobian = np.zeros((3, model.nv))
            angular = np.zeros_like(jacobian)
            foot_velocity = np.zeros(3)
            for foot in feet:
                mujoco.mj_jacGeom(model, data, jacobian, angular, foot)
                foot_velocity += jacobian @ data.qvel / len(feet)
            data.qvel[0] -= foot_velocity[0]
            data.qvel[1] -= foot_velocity[2]
        previous = np.zeros(6)
        lower, upper = model.actuator_ctrlrange[:6].T.copy()
        lower[[0, 1, 3, 4]] *= config.human_strength * config.mt_scale
        upper[[0, 1, 3, 4]] *= config.human_strength * config.mt_scale
        rates_lo = np.array([-500 * config.human_strength * config.mrtd_scale] * 2 + [p.rate_min_nm_s / 2])
        rates_lo = np.tile(rates_lo, 2)
        rates_hi = np.array([500 * config.human_strength * config.mrtd_scale] * 2 + [p.rate_max_nm_s / 2])
        rates_hi = np.tile(rates_hi, 2)
        ankle_joints = [model.joint(f"{side}_ankle").id for side in ["left", "right"]]
    else:
        if config.model == "contact":
            for _ in range(round(0.7 / model.opt.timestep)):
                data.ctrl[:] = 0
                mujoco.mj_step(model, data)
            data.qvel[:] = 0
            data.time = 0
        ankle = model.joint("ankle").id
        qi, vi = model.jnt_qposadr[ankle], model.jnt_dofadr[ankle]
        data.qpos[qi], data.qvel[vi] = config.lean_rad, config.velocity_rad_s
        previous = np.array([config.initial_torque_nm], dtype=float)
    mujoco.mj_forward(model, data)
    foot_names = ["left_sole", "right_sole"] if human else ["foot_-1", "foot_1"]
    initial_foot_x = np.array([data.geom_xpos[model.geom(name).id, 0] for name in foot_names])
    steps = round(config.duration_s / config.timestep_s)
    stride = round(config.control_period_s / config.timestep_s)
    raw, applied_exo = previous.copy(), np.zeros(2 if human else 1)
    held_rate = 0.0
    pending_commands = deque()
    delay_steps = round(config.command_delay_s / config.timestep_s)
    sample_time = -1.0
    initial_state = np.r_[data.qpos, data.qvel].copy()
    records: dict[str, list] = {
        name: []
        for name in [
            "time_s",
            "qpos",
            "qvel",
            "human_raw_nm",
            "human_applied_nm",
            "exo_applied_nm",
            "lean_rad",
            "velocity_rad_s",
            "human_torque_nm",
            "exo_torque_nm",
            "com_x_m",
            "com_z_m",
            "normal_n",
            "friction_n",
            "cop_m",
            "cop_margin_m",
            "friction_margin_n",
            "push_n",
            "foot_pitch_rad",
            "foot_slide_m",
            "command_sample_time_s",
            "command_age_s",
        ]
    }
    if hjb is not None:
        records["human_rate_command_nm_s"] = []
        records["human_rate_applied_nm_s"] = []
    latencies = []
    start = perf_counter()
    for step in range(steps):
        q, velocity = (data.qpos[2], data.qvel[2]) if human else (data.qpos[qi], data.qvel[vi])
        time_s = step * config.timestep_s
        if step % stride == 0:
            tick = perf_counter()
            if hjb is not None:
                requested_rate = hjb.rate(config.duration_s - time_s, q, velocity, previous[0], p)
            elif trajectory is not None:
                requested = trajectory.sample(time_s)
            elif matlab is not None:
                state = np.r_[data.qpos, data.qvel, previous] if human else np.r_[q, velocity, previous]
                requested = matlab.sample(time_s, state)
            elif human:
                requested = feedforward - gain @ np.r_[data.qpos - reference, data.qvel]
            else:
                # Human adapts to current device torque; this is a local controller, not HJB.
                total_desired = (
                    -1.8 * p.stiffness_nm * q - 0.8 * np.sqrt(p.inertia_kg_m2 * p.stiffness_nm) * velocity
                )
                compensation = (
                    0 if config.human_feedback == "fixed" else exo_torque(q, velocity, p, config.exo)
                )
                requested = np.array([total_desired - compensation])
            requested_exo = np.zeros_like(applied_exo)
            if human:
                for i, joint in enumerate(ankle_joints):
                    # Positive anatomical ankle angle shares the paper convention.
                    aq, av = data.qpos[model.jnt_qposadr[joint]], data.qvel[model.jnt_dofadr[joint]]
                    requested_exo[i] = 0.5 * exo_torque(aq, av, p, config.exo)
            else:
                requested_exo[0] = exo_torque(q, velocity, p, config.exo)
            pending_commands.append(
                (
                    step + delay_steps,
                    time_s,
                    requested_rate if hjb is not None else requested.copy(),
                    requested_exo,
                )
            )
            latencies.append(perf_counter() - tick)
        while pending_commands and pending_commands[0][0] <= step:
            _, sample_time, command, applied_exo = pending_commands.popleft()
            if hjb is not None:
                held_rate = command
            else:
                raw = command
        if hjb is None:
            if human:
                previous = np.clip(
                    raw,
                    np.maximum(lower, previous + rates_lo * config.timestep_s),
                    np.minimum(upper, previous + rates_hi * config.timestep_s),
                )
            else:
                previous[0] = limit_human(float(raw[0]), previous[0], config.timestep_s, p)
        if human:
            data.ctrl[:6], data.ctrl[6:] = previous, applied_exo
        else:
            data.ctrl[:] = [previous[0], applied_exo[0]]
        if hjb is not None:
            # Torque is a state: apply tau(t), then advance it with the held rate.
            data.ctrl[0] = previous[0]
            next_raw = previous[0] + held_rate * config.timestep_s
            next_torque = limit_human(next_raw, previous[0], config.timestep_s, p)
        data.xfrc_applied[:] = 0
        push = 0.0
        if human and config.push_start_s <= time_s < config.push_start_s + config.push_duration_s:
            push = config.push_n
            data.xfrc_applied[model.body("trunk").id, 0] = push
        mujoco.mj_forward(model, data)
        if config.model == "paper":
            grf = ground_reaction(q, velocity, float(previous.sum() + applied_exo.sum()), p)
        else:
            normal, friction, cop = _contact_signals(model, data)
            foot_names = ["left_sole", "right_sole"] if human else ["foot_-1", "foot_1"]
            corners = []
            for name in foot_names:
                geom = model.geom(name).id
                half = model.geom_size[geom]
                rotation = data.geom_xmat[geom].reshape(3, 3)
                extent = abs(rotation[0]) @ half
                corners += [data.geom_xpos[geom, 0] - extent, data.geom_xpos[geom, 0] + extent]
            grf = {
                "normal_n": normal,
                "friction_n": friction,
                "cop_m": cop,
                "cop_margin_m": min(cop - min(corners), max(corners) - cop) if normal > 1e-6 else -1.0,
                "friction_margin_n": p.friction * normal - abs(friction),
            }
        com = data.subtree_com[model.body("pelvis" if human else "human").id]
        foot_ids = [model.geom(name).id for name in foot_names]
        foot_pitch = max(
            abs(np.arctan2(data.geom_xmat[i].reshape(3, 3)[0, 2], data.geom_xmat[i].reshape(3, 3)[2, 2]))
            for i in foot_ids
        )
        foot_slide = max(abs(data.geom_xpos[i, 0] - initial_foot_x[j]) for j, i in enumerate(foot_ids))
        row = dict(
            foot_pitch_rad=foot_pitch,
            foot_slide_m=foot_slide,
            time_s=time_s,
            command_sample_time_s=sample_time,
            command_age_s=time_s - sample_time if sample_time >= 0 else 0,
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            human_raw_nm=raw.copy(),
            human_applied_nm=previous.copy(),
            exo_applied_nm=applied_exo.copy(),
            lean_rad=q,
            velocity_rad_s=velocity,
            human_torque_nm=float(previous[[2, 5]].sum() if human else previous[0]),
            exo_torque_nm=float(applied_exo.sum()),
            com_x_m=com[0],
            com_z_m=com[2],
            push_n=push,
            **grf,
        )
        if hjb is not None:
            row["human_rate_command_nm_s"] = held_rate
            row["human_rate_applied_nm_s"] = (next_torque - previous[0]) / config.timestep_s
        for name in records:
            records[name].append(row[name])
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
            raise RuntimeError(f"MuJoCo numerical failure at step {step}")
        if hjb is not None:
            raw[0], previous[0] = next_raw, next_torque
    elapsed = perf_counter() - start
    arrays = {key: np.asarray(value) for key, value in records.items()}
    contact_valid = (
        (arrays["normal_n"] > 0)
        & (arrays["cop_margin_m"] >= 0)
        & (arrays["friction_margin_n"] >= 0)
        & (arrays["foot_pitch_rad"] < 0.035)
        & (arrays["foot_slide_m"] < 0.02)
    )
    final = arrays["time_s"] >= config.duration_s - min(1, config.duration_s)
    recovered = bool(
        np.all(abs(arrays["lean_rad"][final]) < 0.05)
        and np.all(abs(arrays["velocity_rad_s"][final]) < 0.05)
        and np.all(contact_valid)
    )
    if human:
        recovered = recovered and bool(np.min(arrays["com_z_m"]) > 0.6 * arrays["com_z_m"][0])
    metrics = {
        "model": config.model,
        "profile": p.name,
        "exo": config.exo,
        "controller": "hjb"
        if hjb
        else "matlab"
        if matlab
        else "replay"
        if trajectory
        else "stance_lqr"
        if human
        else "adaptive_pd",
        "recovered": recovered,
        "contact_valid_fraction": float(np.mean(contact_valid)),
        "max_abs_lean_rad": float(max(abs(arrays["lean_rad"]))),
        "final_lean_rad": float(data.qpos[2] if human else data.qpos[qi]),
        "final_velocity_rad_s": float(data.qvel[2] if human else data.qvel[vi]),
        "final_human_torque_nm": float(previous[[2, 5]].sum() if human else previous[0]),
        "final_exo_torque_nm": float(applied_exo.sum()),
        "human_torque_rms_nm": float(np.sqrt(np.mean(arrays["human_applied_nm"] ** 2))),
        "control_calls": len(latencies),
        "control_p50_us": float(np.median(latencies) * 1e6),
        "control_p99_us": float(np.percentile(latencies, 99) * 1e6),
        "wall_s": elapsed,
        "simulated_s_per_wall_s": config.duration_s / elapsed,
        "physics_hz": 1 / config.timestep_s,
        "control_hz": 1 / config.control_period_s,
        "command_delay_s": config.command_delay_s,
        "control_deadline_miss_fraction": float(np.mean(np.asarray(latencies) > config.control_period_s)),
        "initial_state": initial_state.tolist(),
        "human_feedback": config.human_feedback,
        "feedback_gain": gain.tolist() if human and not (trajectory or matlab) else None,
        "feedback_reference": reference.tolist() if human else None,
        "feedback_design_timestep_s": config.control_period_s
        / max(1, int(np.ceil(config.control_period_s / 0.001)))
        if human
        else None,
        "max_foot_pitch_rad": float(max(arrays["foot_pitch_rad"])),
        "max_foot_slide_m": float(max(arrays["foot_slide_m"])),
        "human_clipped_fraction": float(
            np.mean(np.any(abs(arrays["human_raw_nm"] - arrays["human_applied_nm"]) > 1e-9, axis=1))
        ),
        "total_mass_kg": float(sum(model.body_mass)),
        "effective_profile": asdict(p),
        "controller_source": (
            {"function": matlab.function, "directory": str(matlab.directory)}
            if matlab
            else {"file": str(hjb.path), "sha256": hjb.sha256}
            if hjb
            else {"interpolation": trajectory.interpolation, "outside": trajectory.outside}
            if trajectory
            else None
        ),
        "scope": "controller-specific rollout; not an HJB stabilizable region",
    }
    samples = {"time_s": trajectory.time_s, "torque_nm": trajectory.values} if trajectory else None
    return Episode(config, xml, arrays, metrics, samples)
