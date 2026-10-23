"""GPU integration of MATLAB-prepared HJB fields, with streamed policy output.

The physics, targets and obstacles come from solve_paper_hjb(prepare_only=true).
This is an independent numerical backend, not the authors' MATLAB solver.
Requires the separate Python 3.12 environment-hjb.yml environment.
"""

import argparse
import hashlib
import json
import shutil
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import h5py
import hj_reachability as hj
import jax
import jax.numpy as jnp
import numpy as np
from hj_reachability.finite_differences.upwind_first import WENO5, _diff_coefficients, _unrolled_correlate
from scipy.io import savemat

jax.config.update("jax_enable_x64", True)

AXIS_NAMES = ("theta_rad", "velocity_rad_s", "human_torque_nm", "time_to_go_s")


def _eno3(values, spacing, boundary_condition):
    """ENO3 with divided-difference comparisons and right-biased exact ties.

    Adapted from StanfordASL/hj_reachability's ENO implementation (MIT;
    Copyright 2021 Ed Schmerling). See third_party/hj_reachability_LICENSE.
    The tie rule and scaled differences match ToolboxLS's ENO3a convention.
    """
    values = boundary_condition(values, 3)
    first = (1 / spacing) * (values[1:] - values[:-1])
    candidates = tuple(
        _unrolled_correlate(first[i : len(first) - 3 + i], coefficients)
        for i, coefficients in enumerate(_diff_coefficients(3))
    )
    second = (0.5 / spacing) * (first[1:] - first[:-1])
    third = ((1 / 3) / spacing) * (second[1:] - second[:-1])
    right = jnp.abs(third[1:]) <= jnp.abs(third[:-1])
    second = second[1:-1]
    index = jnp.where(jnp.abs(second[1:]) <= jnp.abs(second[:-1]), right[1:] + 1, right[:-1])
    return (
        jnp.select([index[:-1] == i for i in range(2)], candidates[:-2], candidates[-2]),
        jnp.select([index[1:] == i for i in range(2)], candidates[1:-1], candidates[-1]),
    )


def _godunov_control(left, right, rate_bounds):
    """Exact upwind Hamiltonian for V_t = min_u u V_tau.

    For a concave Hamiltonian, use its maximum on [D-, D+] when
    D- <= D+, otherwise its minimum on [D+, D-]. The only interior
    extremum is at zero. This also handles one-sided rate intervals.
    """
    h_left = jnp.minimum(rate_bounds[0] * left, rate_bounds[1] * left)
    h_right = jnp.minimum(rate_bounds[0] * right, rate_bounds[1] * right)
    maximum = jnp.maximum(h_left, h_right)
    maximum = jnp.where((left <= 0) & (right >= 0), jnp.maximum(maximum, 0), maximum)
    return jnp.where(left <= right, maximum, jnp.minimum(h_left, h_right))


def _rhs(value, grid, drift, rate_bounds, flux="lax_friedrichs"):
    left, right = grid.upwind_grad_values(WENO5, value)
    if flux == "godunov":
        # Drift is linear in the gradient, so its LF flux already equals upwind.
        advection = jnp.sum(drift * jnp.where(drift >= 0, right, left), axis=-1)
        control = _godunov_control(left[..., 2], right[..., 2], rate_bounds)
        speeds = jnp.max(jnp.abs(drift), axis=(0, 1, 2)).at[2].set(jnp.max(jnp.abs(rate_bounds)))
        return advection + control, 1 / jnp.sum(speeds / jnp.array(grid.spacings))
    gradient = (left + right) / 2
    control = jnp.where(gradient[..., 2] >= 0, rate_bounds[0], rate_bounds[1])
    lower = jnp.minimum(jnp.min(left[..., 2]), jnp.min(right[..., 2]))
    upper = jnp.maximum(jnp.max(left[..., 2]), jnp.max(right[..., 2]))
    alpha_rate = jnp.maximum(
        jnp.abs(jnp.where(lower >= 0, rate_bounds[0], rate_bounds[1])),
        jnp.abs(jnp.where(upper >= 0, rate_bounds[0], rate_bounds[1])),
    )
    alpha = jnp.abs(drift).at[..., 2].set(alpha_rate)
    hamiltonian = jnp.sum(gradient * drift, axis=-1) + gradient[..., 2] * control
    dissipation = jnp.sum(alpha * (right - left), axis=-1) / 2
    # ToolboxLS sums each axis's global maximum, rather than taking max(sum()).
    bound = 1 / jnp.sum(jnp.max(alpha, axis=(0, 1, 2)) / jnp.array(grid.spacings))
    return hamiltonian + dissipation, bound


@partial(jax.jit, static_argnames=("flux",))
def advance(
    value, start, end, grid, drift, obstacle, rate_bounds, initial, tube, cfl=0.8, flux="lax_friedrichs"
):
    """One saved interval with SSP-RK3, CFL stepping and contact obstacles.

    Godunov stages respect the terminal cost's global lower bound. The exact
    reach-avoid value cannot be smaller than min(initial); enforcing that
    bound prevents ENO interpolation from inventing deeper negative minima.
    The historical LF path is unchanged for reproduction.
    """
    floor = jnp.min(initial)

    def bounded(state):
        return jnp.maximum(state, floor) if flux == "godunov" else state

    def step(carry):
        current, state = carry
        first, bound = _rhs(state, grid, drift, rate_bounds, flux)
        dt = jnp.minimum(cfl * bound, end - current)
        first_state = bounded(state + dt * first)
        second, _ = _rhs(first_state, grid, drift, rate_bounds, flux)
        half_state = bounded(0.25 * (3 * state + first_state + dt * second))
        third, _ = _rhs(half_state, grid, drift, rate_bounds, flux)
        final = bounded((state + 2 * (half_state + dt * third)) / 3)
        final = jnp.where(tube, jnp.minimum(final, initial), final)
        return current + dt, jnp.maximum(final, obstacle)

    # Match helperOC's 1e-4 nondimensional end-of-interval stopping tolerance.
    return jax.lax.while_loop(lambda carry: carry[0] < end - 1e-4, step, (start, value))[1]


def _decode(dataset) -> str:
    return "".join(chr(int(code)) for code in np.asarray(dataset).ravel())


def solve(
    problem: Path,
    output: Path,
    save_history: bool = True,
    platform: str = "gpu",
    cfl: float = 0.8,
    flux: str = "lax_friedrichs",
) -> dict:
    """Integrate one prepared problem, preserving fields and physical contracts."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an HJB run: {output}")
    if not np.isfinite(cfl) or not 0 < cfl <= 0.8:
        raise ValueError("CFL must be in (0,0.8]")
    if flux not in {"lax_friedrichs", "godunov"}:
        raise ValueError("Unknown numerical flux")
    devices = jax.devices(platform)
    with jax.default_device(devices[0]), h5py.File(problem, "r") as source:
        metadata = json.loads(_decode(source["problem_json"]))
        summary = metadata["summary"]
        axes = [np.asarray(source[name]).ravel() for name in AXIS_NAMES]
        shape = tuple(len(axis) for axis in axes[:3])
        if any(len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0) for axis in axes):
            raise ValueError("Prepared problem requires finite, increasing axes")
        scale = np.asarray(source["state_scale"]).ravel()
        time_scale = float(np.asarray(source["time_scale_s"]).item())
        state_axes = [axis * factor for axis, factor in zip(axes[:3], scale, strict=True)]
        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(jnp.array([a[0] for a in state_axes]), jnp.array([a[-1] for a in state_axes])), shape
        )
        arrays = {
            name: np.asarray(source[name], dtype=float).T
            for name in ["initial_value", "obstacle_value", "drift_theta", "drift_velocity"]
        }
        if any(array.shape != shape or not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("Prepared fields must be finite and match the spatial axes")
        initial = jnp.array(arrays["initial_value"])
        obstacle = jnp.array(arrays["obstacle_value"])
        drift = jnp.stack(
            [jnp.array(arrays["drift_theta"]), jnp.array(arrays["drift_velocity"]), jnp.zeros(shape)], axis=-1
        )
        rate_bounds = jnp.array(np.asarray(source["rate_bounds"]).ravel())
        if summary["method"] not in {"none", "minVWithV0"}:
            raise ValueError("Only the explicit set and tube methods are supported")
        tube = summary["method"] == "minVWithV0"
        times = axes[3] / time_scale
        args = (initial, times[0], times[1], grid, drift, obstacle, rate_bounds, initial, tube, cfl)
        started = time.perf_counter()
        compiled = advance.lower(*args, flux=flux).compile()
        compilation_s = time.perf_counter() - started
        memory = compiled.memory_analysis()
        statistics = devices[0].memory_stats()
        if statistics is not None:
            required = statistics["bytes_in_use"] + memory.temp_size_in_bytes + memory.output_size_in_bytes
            if required > 0.9 * statistics["bytes_limit"]:
                raise MemoryError(
                    f"Compiled HJB step needs approximately {required} device bytes; run stopped"
                )
        output.mkdir(parents=True)
        shutil.copyfile(__file__, output / "source.py")
        with problem.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        summary = dict(
            summary,
            stage="running",
            backend="JAX/WENO5",
            device=str(devices[0]),
            device_model=devices[0].device_kind,
            solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            precision="float64",
            problem_sha256=digest,
            save_history=save_history,
            cfl=cfl,
            flux=flux,
            compilation_s=compilation_s,
            compiled_temporary_bytes=memory.temp_size_in_bytes,
            versions={name: version(name) for name in ["jax", "jaxlib", "hj_reachability", "numpy", "h5py"]},
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        partial_policy = output / "policy.partial.mat"
        policy = h5py.File(partial_policy, "w") if save_history else None
        try:
            if policy is not None:
                for name, axis in zip(AXIS_NAMES, axes, strict=True):
                    policy[name] = axis[None, :]
                policy["contract_json"] = np.array(
                    [ord(c) for c in json.dumps(metadata["contract"])], dtype=np.uint16
                )[:, None]
                history = policy.create_dataset(
                    "value",
                    (len(times), *shape[::-1]),
                    dtype="f8",
                    chunks=(1, shape[2], 1, shape[0]),
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                )
                history[0] = arrays["initial_value"].T
            value = initial
            started = time.perf_counter()
            with (output / "progress.jsonl").open("w") as progress:
                for index in range(1, len(times)):
                    value = compiled(
                        value,
                        times[index - 1],
                        times[index],
                        grid,
                        drift,
                        obstacle,
                        rate_bounds,
                        initial,
                        tube,
                        cfl,
                    )
                    actual = np.asarray(value)
                    if not np.isfinite(actual).all():
                        raise FloatingPointError(f"Nonfinite HJB field at {axes[3][index]} seconds")
                    if policy is not None:
                        history[index] = actual.T
                    record = {"time_s": float(axes[3][index]), "wall_s": time.perf_counter() - started}
                    progress.write(json.dumps(record) + "\n")
                    progress.flush()
                    print(json.dumps(record), flush=True)
            summary.update(
                stage="solved",
                wall_s=time.perf_counter() - started,
                final_reachable_nodes=int(np.count_nonzero(actual <= 0)),
            )
            savemat(
                output / "projection.mat",
                {
                    "theta_rad": axes[0],
                    "velocity_rad_s": axes[1],
                    "projected_value": actual.min(axis=2),
                },
            )
            np.save(output / "final_value.npy", actual)
        finally:
            if policy is not None:
                policy.close()
        if policy is not None:
            partial_policy.rename(output / "policy.mat")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True, help="MATLAB-prepared problem.mat")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--projection-only", action="store_true", help="Save final fields, without feedback history"
    )
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--flux", choices=["lax_friedrichs", "godunov"], default="lax_friedrichs")
    args = parser.parse_args()
    print(
        json.dumps(
            solve(args.problem, args.out, not args.projection_only, args.platform, args.cfl, args.flux),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
