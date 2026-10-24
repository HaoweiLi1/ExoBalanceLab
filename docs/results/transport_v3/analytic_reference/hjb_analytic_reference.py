"""Run a thin-box double-integrator reference with an analytic reachable set.

State is position [m], velocity [m/s]; acceleration lies in [-1, 1] m/s^2.
This isolates numerical transport from anthropometry, contact and MATLAB targets.
Use the separate JAX environment; outputs are scientific diagnostics, not unit tests.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator


def exact_bounds(velocity, duration=0.5, position_width=0.005, velocity_width=0.01):
    """Return initial-position bounds and feasible velocities for the terminal box.

    W is the integrated acceleration. At fixed W, the extrema of
    integral((T-t)*u(t), t=0..T) use one bang-bang switch. Optimizing W over
    the terminal-velocity interval gives these exact finite-horizon bounds.
    """
    lower_impulse = np.maximum(-duration, -velocity_width - velocity)
    upper_impulse = np.minimum(duration, velocity_width - velocity)
    minimum = -(duration**2) / 4 + duration * lower_impulse / 2 + lower_impulse**2 / 4
    maximum = duration**2 / 4 + duration * upper_impulse / 2 - upper_impulse**2 / 4
    lower = -position_width - velocity * duration - maximum
    upper = position_width - velocity * duration - minimum
    return lower, upper, np.abs(velocity) <= duration + velocity_width


def run(output, resolutions, platform):
    import hj_reachability as hj
    import jax
    import jax.numpy as jnp

    from exobalancelab import hjb_solver

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "solver.py").write_bytes(Path(hjb_solver.__file__).read_bytes())
    (output / "benchmark.py").write_bytes(Path(__file__).read_bytes())
    axes = [np.linspace(-0.25, 0.25, 601), np.linspace(-0.75, 0.75, 601)]
    position, velocity = np.meshgrid(*axes, indexing="ij")
    lower, upper, feasible = exact_bounds(velocity)
    exact = feasible & (position >= lower) & (position <= upper)

    def area(mask):
        return float(np.trapezoid(np.trapezoid(mask.astype(float), axes[1], axis=1), axes[0]))

    continuous_area = quad(
        lambda speed: float(exact_bounds(speed)[1] - exact_bounds(speed)[0]),
        -0.51,
        0.51,
        points=[-0.49, 0.49],
        epsabs=1e-12,
    )[0]
    metadata = {
        "duration_s": 0.5,
        "position_half_width_m": 0.005,
        "velocity_half_width_m_s": 0.01,
        "acceleration_bounds_m_s2": [-1, 1],
        "resolutions": resolutions,
        "exact_continuous_area": continuous_area,
        "exact_common_grid_area": area(exact),
        "solver_sha256": hashlib.sha256(Path(hjb_solver.__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison": "Absolute error against an independent analytic set; not the paper's spatial gate.",
    }
    rows = []
    with jax.default_device(jax.devices(platform)[0]):
        for n in resolutions:
            grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
                hj.sets.Box(jnp.array([-1.0, -0.25, -0.75]), jnp.array([1.0, 0.25, 0.75])), (3, n, n)
            )
            initial = 0.005 * (
                jnp.maximum(jnp.abs(grid.states[..., 1]) / 0.005, jnp.abs(grid.states[..., 2]) / 0.01) - 1
            )
            drift = jnp.zeros_like(grid.states).at[..., 1].set(grid.states[..., 2])
            obstacle = jnp.full(initial.shape, -100.0)
            rates = jnp.array([-1.0, 1.0])
            coordinates = [np.asarray(axis) for axis in grid.coordinate_vectors[1:]]
            for flux in ["lax_friedrichs", "godunov"]:
                value = initial
                started = time.perf_counter()
                for k in range(1, 51):
                    value = hjb_solver.advance(
                        value,
                        (k - 1) * 0.01,
                        k * 0.01,
                        grid,
                        drift,
                        obstacle,
                        rates,
                        initial,
                        False,
                        flux=flux,
                    )
                field = np.asarray(value)[1]
                mask = RegularGridInterpolator(coordinates, field)(np.stack([position, velocity], -1)) <= 0
                name = f"{flux}_{n}.npz"
                np.savez_compressed(
                    output / name, position=coordinates[0], velocity=coordinates[1], value=field
                )
                row = {
                    "resolution": n,
                    "flux": flux,
                    "field": name,
                    "area": area(mask),
                    "area_error_percent": 100 * abs(area(mask) - area(exact)) / area(exact),
                    "iou": float(np.count_nonzero(mask & exact) / np.count_nonzero(mask | exact)),
                    "wall_s": time.perf_counter() - started,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                (output / "summary.json").write_text(json.dumps(dict(metadata, cases=rows), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[101, 201, 401])
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    run(args.out, args.resolutions, args.platform)
