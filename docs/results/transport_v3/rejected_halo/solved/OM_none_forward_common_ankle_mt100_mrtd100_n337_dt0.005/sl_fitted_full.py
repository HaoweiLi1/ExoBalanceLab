"""Finite-horizon reach-avoid transport on an enclosing physical grid.

Exact split flows, Strang composition and bounded ENO interpolation preserve the
frozen terminal boxes, torque/rate bounds and paper contact constraints. This
backend exports convergence fields; the legacy policy solver remains available.
"""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import h5py
import jax
import jax.numpy as j
import numpy as np
from flax import struct
from scipy.io import savemat

from exobalancelab import paper
from exobalancelab.paper import Profile

jax.config.update("jax_enable_x64", True)


@struct.dataclass
class Problem:
    """Physical domain and dynamics; arrays use SI units throughout."""

    lower: object
    upper: object
    parameters: object
    rates: object
    kind: str = struct.field(pytree_node=False, default="paper")


def source_hashes():
    """Hash the solver and the profile definitions that supply its physics."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(paper.__file__)]
    }


def load_problem(source):
    """Read a MATLAB preparation or its compact, lossless physical input."""
    if source.suffix == ".json":
        return json.loads(source.read_text())
    with h5py.File(source) as f:
        raw = "".join(chr(int(c)) for c in np.asarray(f["problem_json"]).ravel())
        summary = json.loads(raw)["summary"]
        axes = [np.asarray(f[k]).ravel() for k in ["theta_rad", "velocity_rad_s", "human_torque_nm"]]
        scale = np.asarray(f["state_scale"]).ravel().tolist()
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(
        summary=summary,
        lower=[float(a[0]) for a in axes],
        upper=[float(a[-1]) for a in axes],
        state_scale=scale,
        prepared_sha256=digest,
    )


def paper_problem(summary, lower, upper):
    """Use the frozen effective MT/MRTD parameters without resolving the goal."""
    p = Profile(name=summary["profile"], **summary["effective_profile"])
    if summary["exo"] not in {"none", "gc"} or summary["method"] != "none":
        raise ValueError("This transport campaign supports finite-horizon none/GC sets")
    if not (0 < lower[0] < upper[0] < np.pi) or p.friction != 1:
        raise ValueError("The split-flow enclosure requires the paper's angle domain and friction")
    parameters = [
        p.body_mass_kg,
        p.mass_kg - p.body_mass_kg,
        p.length_m,
        p.gravity_m_s2,
        p.damping_nm_s,
        p.foot_length_m,
        0.19 * p.foot_length_m,
        0.039 * p.height_m,
        0.31 * p.foot_length_m,
        float(summary["exo"] == "gc"),
        p.torque_min_nm,
        p.torque_max_nm,
    ]
    return Problem(*map(j.asarray, [lower, upper, parameters, [p.rate_min_nm_s, p.rate_max_nm_s]]))


def analytic_problem(cap=np.inf):
    """Double integrator in the same three-axis kernel; axis zero is passive."""
    return Problem(
        *map(j.asarray, [[-1, -0.25, -0.75], [1, 0.25, 0.75], [cap], [-1, 1]]), kind="double_integrator"
    )


def physical_fields(problem, states):
    """Return uncontrolled drift and the original C1/C3-C5 obstacle field."""
    theta, velocity, torque = states
    if problem.kind == "double_integrator":
        return (j.zeros_like(theta), torque), -j.inf
    mass, foot_mass, length, gravity, damping, foot_length, ankle, height, foot_com, gc = problem.parameters[
        :10
    ]
    q = theta - j.pi / 2
    stiffness = mass * gravity * length
    device = j.clip(-stiffness * j.sin(q), -50, 50) * gc
    total = torque + device
    acceleration = (stiffness * j.sin(q) - damping * velocity + total) / (mass * length**2)
    fx = mass * length * (j.cos(q) * acceleration - j.sin(q) * velocity**2)
    fz = (mass + foot_mass) * gravity - mass * length * (j.sin(q) * acceleration + j.cos(q) * velocity**2)
    moment = -total + damping * velocity + foot_mass * gravity * foot_com - height * fx
    weight = (mass + foot_mass) * gravity
    margin = j.minimum(fz / weight, (moment + ankle * fz) / (weight * foot_length))
    margin = j.minimum(margin, ((foot_length - ankle) * fz - moment) / (weight * foot_length))
    margin = j.minimum(margin, (fz - j.abs(fx)) / weight)
    margin = j.minimum(margin, (torque - problem.parameters[10]) / stiffness)
    margin = j.minimum(margin, (problem.parameters[11] - torque) / stiffness)
    return (velocity, acceleration), -margin


def states(axes):
    return [axis.reshape(tuple(len(axis) if k == d else 1 for k in range(3))) for d, axis in enumerate(axes)]


def interpolate(y, q, axis, extend_lower, extend_upper):
    y = j.moveaxis(y, axis, -1)
    q = j.moveaxis(q, axis, -1)
    i = j.floor(q).astype(j.int32)
    t = q - i
    n = y.shape[-1]

    def sample(k):
        idx = j.clip(i + k, 0, n - 1)
        result = j.take_along_axis(y, idx, axis=-1)
        lo = y[..., :1]
        hi = y[..., -1:]
        lower = lo - (i + k) * j.abs(y[..., 1:2] - lo) * j.sign(lo)
        upper = hi + (i + k - n + 1) * j.abs(hi - y[..., -2:-1]) * j.sign(hi)
        return j.where(i + k < 0, lower, j.where(i + k > n - 1, upper, result))

    a, b, c, d, e, f = [sample(k) for k in [-2, -1, 0, 1, 2, 3]]
    dl = b - 2 * c + d
    dr = c - 2 * d + e
    left = j.abs(dl) < j.abs(dr)
    second = j.where(left, dl, dr)
    tl = d - 3 * c + 3 * b - a
    tc = e - 3 * d + 3 * c - b
    tr = f - 3 * e + 3 * d - c
    third = j.where(left, j.where(j.abs(tl) < j.abs(tc), tl, tc), j.where(j.abs(tc) < j.abs(tr), tc, tr))
    value = (
        c + t * (d - c) + 0.5 * t * (t - 1) * second + t * (t - 1) * j.where(left, t + 1, t - 2) * third / 6
    )
    value = j.where((q >= 0) & (q <= n - 1), j.clip(value, j.minimum(c, d), j.maximum(c, d)), value)
    # A newly requested state outside a still-expanding enclosure is provably
    # outside the previous reachable set. Keep its extension continuous and
    # positive: an exterior jump contaminates ENO and erodes thin sets.
    lo = y[..., :1]
    hi = y[..., -1:]
    value = j.where((q < 0) & extend_lower, j.abs(lo) - q * j.abs(y[..., 1:2] - lo), value)
    value = j.where((q > n - 1) & extend_upper, j.abs(hi) + (q - n + 1) * j.abs(hi - y[..., -2:-1]), value)
    return j.moveaxis(value, -1, axis)


def interval_minimum(value, lower, upper):
    n = value.shape[-1]
    lo = j.maximum(j.ceil(lower).astype(j.int32), 0)
    hi = j.minimum(j.floor(upper).astype(j.int32), n - 1)
    count = j.maximum(hi - lo + 1, 1)
    levels = j.floor(j.log2(count)).astype(j.int32)
    positions = j.arange(n)

    def body(carry):
        level, window, answer = carry
        width = 1 << level
        a = j.take(window, j.clip(lo, 0, n - 1), axis=-1)
        b = j.take(window, j.clip(hi - width + 1, 0, n - 1), axis=-1)
        answer = j.where(levels == level, j.minimum(a, b), answer)
        following = j.take(window, j.minimum(positions + width, n - 1), axis=-1)
        following = j.where(positions + width < n, following, j.inf)
        return level + 1, j.minimum(window, following), answer

    _, _, answer = jax.lax.while_loop(
        lambda c: c[0] <= j.max(levels), body, (0, value, j.full(value.shape, j.inf))
    )
    return j.where(lo <= hi, answer, j.inf)


def project(y, axes, p):
    x = states(axes)
    if p.kind in {"double_integrator", "triple_integrator"}:
        obstacle = j.abs(x[2]) - p.parameters[0]
    else:
        _, obstacle = physical_fields(p, x)
    return j.maximum(y, obstacle)


def control(y, axes, p, dt):
    tau = axes[2]
    n = len(tau)
    h = (tau[-1] - tau[0]) / (n - 1)
    next_tau = j.linspace(
        j.maximum(p.lower[2], tau[0] - p.rates[1] * dt), j.minimum(p.upper[2], tau[-1] - p.rates[0] * dt), n
    )
    lo = (next_tau + p.rates[0] * dt - tau[0]) / h
    hi = (next_tau + p.rates[1] * dt - tau[0]) / h
    low_query = j.broadcast_to(lo, y.shape)
    high_query = j.broadcast_to(hi, y.shape)
    edge = (tau[0] > p.lower[2], tau[-1] < p.upper[2])
    result = j.minimum(
        interval_minimum(y, lo, hi),
        j.minimum(interpolate(y, low_query, 2, *edge), interpolate(y, high_query, 2, *edge)),
    )
    axes = (axes[0], axes[1], next_tau)
    return project(result, axes, p), axes


def angle(y, axes, p, dt):
    if p.kind == "double_integrator":
        return y, axes
    q, v, tau = axes
    n = len(q)
    h = (q[-1] - q[0]) / (n - 1)
    next_q = j.linspace(j.maximum(p.lower[0], q[0] - v[-1] * dt), j.minimum(p.upper[0], q[-1] - v[0] * dt), n)
    query = j.broadcast_to((next_q[:, None, None] + v[None, :, None] * dt - q[0]) / h, y.shape)
    result = interpolate(y, query, 0, q[0] > p.lower[0], q[-1] < p.upper[0])
    axes = (next_q, v, tau)
    return project(result, axes, p), axes


def velocity(y, axes, p, dt):
    q, v, tau = axes
    n = len(v)
    h = (v[-1] - v[0]) / (n - 1)
    if p.kind in {"double_integrator", "triple_integrator"}:
        constant = tau[None, None, :]
        minimum = tau[0]
        maximum = tau[-1]
        decay = 1.0
        gain = dt
    else:
        mass, _, length, gravity, damping, *_ = p.parameters
        stiffness = mass * gravity * length
        inertia = mass * length**2
        gc = p.parameters[9]

        def restoring(theta):
            human = stiffness * j.sin(theta - j.pi / 2)
            return human - j.clip(human, -50, 50) * gc

        constant = (restoring(q[:, None, None]) + tau[None, None, :]) / inertia
        minimum = (restoring(q[0]) + tau[0]) / inertia
        maximum = (restoring(q[-1]) + tau[-1]) / inertia
        rate = damping / inertia
        decay = j.exp(-rate * dt)
        gain = -j.expm1(-rate * dt) / rate
    next_v = j.linspace(
        j.maximum(p.lower[1], (v[0] - maximum * gain) / decay),
        j.minimum(p.upper[1], (v[-1] - minimum * gain) / decay),
        n,
    )
    query = j.broadcast_to((next_v[None, :, None] * decay + constant * gain - v[0]) / h, y.shape)
    result = interpolate(y, query, 1, v[0] > p.lower[1], v[-1] < p.upper[1])
    axes = (q, next_v, tau)
    return project(result, axes, p), axes


@jax.jit
def step(value, axes, p, dt):
    value, axes = control(value, axes, p, dt / 2)
    value, axes = angle(value, axes, p, dt / 2)
    value, axes = velocity(value, axes, p, dt)
    value, axes = angle(value, axes, p, dt / 2)
    return control(value, axes, p, dt / 2)


def integrate(p, boxes, shape, duration, time_step, progress=None):
    padding=np.array([(n-1)/(n-3) if n>3 else 2 for n in shape])
    lower = np.maximum(np.asarray(p.lower), np.min([c - padding * w for c, w, d in boxes], axis=0))
    upper = np.minimum(np.asarray(p.upper), np.max([c + padding * w for c, w, d in boxes], axis=0))
    axes = tuple(j.linspace(lo, hi, n) for lo, hi, n in zip(lower, upper, shape))
    value = j.full(shape, j.inf)
    for center, widths, depth in boxes:
        cost = -j.ones(shape)
        for axis, state in enumerate(states(axes)):
            if p.kind == "double_integrator" and axis == 0:
                continue
            cost = j.maximum(cost, j.abs((state - center[axis]) / widths[axis]) - 1)
        value = j.minimum(value, depth * cost)
    value = project(value, axes, p)
    if duration <= 0 or time_step <= 0 or min(shape) < 3:
        raise ValueError("Positive horizon/time step and at least three nodes are required")
    count = int(np.ceil(duration / time_step))
    dt = duration / count
    compiled = step.lower(value, axes, p, dt).compile()
    memory = compiled.memory_analysis()
    stats = value.devices().pop().memory_stats()
    if stats is not None:
        required = stats["bytes_in_use"] + memory.temp_size_in_bytes + memory.output_size_in_bytes
        if required > 0.9 * stats["bytes_limit"]:
            raise MemoryError(f"Preflight: {required} device bytes needed")
    for k in range(count):
        value, axes = compiled(value, axes, p, dt)
        if progress and (k + 1) % max(1, count // 14) == 0:
            progress((k + 1) * dt)
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise FloatingPointError("Nonfinite solution")
    return [np.asarray(a) for a in axes], value


def reference(n, time_step, cap=np.inf):
    """Solve the independently specified thin-box benchmark, including velocity limits."""
    return integrate(
        analytic_problem(cap),
        [(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.005, 0.01]), 0.005)],
        (3, n, n),
        0.5,
        time_step,
    )


def solve(source, output, resolution=225, time_step=0.01, projection_only=False):
    """Export a frozen-input solve with actual geometry and complete source identity."""
    if output.exists():
        raise FileExistsError(output)
    if resolution < 5 or not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("At least five nodes and a positive finite time step are required")
    inputs = load_problem(source)
    summary = inputs["summary"]
    scale = np.asarray(inputs["state_scale"])
    p = paper_problem(summary, np.array(inputs["lower"]), np.array(inputs["upper"]))
    boxes = []
    for label, key in [("Ank", "ankleEq"), ("Toe", "toeEq")]:
        center = np.array(summary["target"][key])
        widths = np.array([summary["target"][f"{a}{label}Range"] for a in ["theta", "thetaDot", "torque"]])
        box = (center, widths, min(widths * scale))
        if not any(np.array_equal(center, old[0]) and np.array_equal(widths, old[1]) for old in boxes):
            boxes.append(box)
    output.mkdir(parents=True)
    physical_input = json.dumps(inputs, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (output / "problem.json").write_text(physical_input)
    sources = source_hashes()
    for filename in sources:
        (output / filename).write_bytes((Path(__file__).parent / filename).read_bytes())
    keys = [
        "profile",
        "exo",
        "direction",
        "target",
        "effective_profile",
        "mt_scale",
        "mrtd_scale",
        "target_definition",
        "target_semantics",
        "method",
        "duration_s",
        "gravity_m_s2",
    ]
    count = int(np.ceil(summary["duration_s"] / time_step))
    metadata = {k: summary[k] for k in keys}
    metadata.update(
        stage="running",
        backend="enclosing-grid/Strang/SL-ENO",
        precision="float64",
        resolution=[resolution] * 3,
        requested_time_step_s=time_step,
        time_step_s=summary["duration_s"] / count,
        integration_steps=count,
        sources=sources,
        input_sha256=hashlib.sha256(physical_input.encode()).hexdigest(),
        prepared_sha256=inputs["prepared_sha256"],
        versions={name: version(name) for name in ["jax", "jaxlib", "numpy", "flax"]},
        device=str(jax.devices()[0]),
        output_scope="Convergence projection; no feedback history exported",
    )
    started = time.perf_counter()

    def save_summary():
        (output / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    save_summary()

    def progress(t):
        print(json.dumps(dict(time_s=t, wall_s=time.perf_counter() - started)), flush=True)

    # Infimum over controls and the reach-avoid obstacle commute with the
    # minimum over terminal goals. Propagate each box on its own tight grid;
    # a shared initial domain can miss a narrow ankle goal between torque nodes.
    value = None
    for index, box in enumerate(boxes):
        print(json.dumps(dict(terminal_box=index)), flush=True)
        axes, actual = integrate(p, [box], (resolution,) * 3, summary["duration_s"], time_step, progress)
        if not all(
            np.allclose(a[[0, -1]], [lo, hi], atol=1e-12, rtol=0)
            for a, lo, hi in zip(axes, inputs["lower"], inputs["upper"])
        ):
            raise ValueError("The final enclosure has not reached the original physical domain")
        value = actual if value is None else np.minimum(value, actual)
    metadata["independently_resolved_boxes"] = len(boxes)
    if not projection_only:
        np.save(output / "final_value.npy", value)
    savemat(
        output / "projection.mat",
        dict(theta_rad=axes[0], velocity_rad_s=axes[1], projected_value=value.min(axis=2)),
    )
    metadata.update(
        stage="solved",
        wall_s=time.perf_counter() - started,
        reachable_nodes=int((value <= 0).sum()),
        projection_sha256=hashlib.sha256((output / "projection.mat").read_bytes()).hexdigest(),
    )
    save_summary()
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=225)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--projection-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(solve(args.problem, args.out, args.resolution, args.step, args.projection_only), indent=2)
    )
