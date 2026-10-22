"""Reproducible numerical validation and transport timing; no HJB claims from rollouts."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.integrate import solve_ivp

from exobalancelab.bridge import Trajectory
from exobalancelab.models import compile_model, paper_xml
from exobalancelab.paper import PROFILES


def validate_dynamics(seed: int = 240805418) -> dict:
    """Compare 1,500 MuJoCo accelerations and 15 integrations to independent Eq. (3)."""
    rng = np.random.default_rng(seed)
    results = {}
    for name, p in PROFILES.items():
        model, data = compile_model(paper_xml(p))
        errors = []
        for _ in range(300):
            q, speed = rng.uniform(-0.3, 0.3), rng.uniform(-1, 1)
            human, exo = rng.uniform(p.torque_min_nm, p.torque_max_nm), rng.uniform(-50, 50)
            data.qpos[0], data.qvel[0], data.ctrl[:] = q, speed, [human, exo]
            mujoco.mj_forward(model, data)
            expected = (
                -9.81 / p.length_m * np.cos(np.pi / 2 + q)
                + (human + exo - p.damping_nm_s * speed) / p.inertia_kg_m2
            )
            errors.append(abs(data.qacc[0] - expected))

        def ode(t, x):
            return [x[1], 9.81 / p.length_m * np.sin(x[0]) + (-10 - p.damping_nm_s * x[1]) / p.inertia_kg_m2]

        reference = solve_ivp(ode, [0, 0.5], [-0.03, 0.2], rtol=1e-11, atol=1e-13).y[:, -1]
        integrations = []
        for dt in [0.002, 0.001, 0.0005]:
            model, data = compile_model(paper_xml(p, dt))
            data.qpos[0], data.qvel[0], data.ctrl[0] = -0.03, 0.2, -10
            for _ in range(round(0.5 / dt)):
                mujoco.mj_step(model, data)
            difference = np.r_[data.qpos, data.qvel] - reference
            integrations.append(
                {
                    "dt_s": dt,
                    "angle_error_rad": abs(difference[0]),
                    "velocity_error_rad_s": abs(difference[1]),
                }
            )
        results[name] = {
            "profile": asdict(p),
            "max_acceleration_error_rad_s2": max(errors),
            "sample_count": len(errors),
            "integrations": integrations,
        }
    return {"seed": seed, "mujoco_version": mujoco.__version__, "profiles": results}


def benchmark_bridge(seed: int = 240805418) -> dict:
    """Time old nearest-neighbour scan vs new interpolation on non-grid query times.

    The two algorithms intentionally differ: legacy nearest-neighbour versus continuous
    linear interpolation. This comparison measures lookup cost, not MATLAB Engine IPC.
    """
    samples = 12001
    times = np.linspace(0, 6, samples)
    values = np.sin(times)
    trajectory = Trajectory(times, values)
    table = dict(zip(times, values))
    queries = np.random.default_rng(seed).uniform(0, 6, 1000)
    start = perf_counter()
    old = [table[min(table, key=lambda item: abs(item - time))] for time in queries]
    old_s = perf_counter() - start
    start = perf_counter()
    new = [trajectory.sample(time)[0] for time in queries]
    new_s = perf_counter() - start
    return {
        "seed": seed,
        "samples": samples,
        "queries": len(queries),
        "legacy_scan_us_per_query": old_s / len(queries) * 1e6,
        "interpolation_us_per_query": new_s / len(queries) * 1e6,
        "speedup": old_s / new_s,
        "legacy_max_sine_error": float(max(abs(np.asarray(old) - np.sin(queries)))),
        "interpolation_max_sine_error": float(max(abs(np.asarray(new) - np.sin(queries)))),
        "scope": "in-process off-grid trajectory lookup; excludes MATLAB execution and physics",
    }


def main() -> None:
    """Write fresh validation evidence to the requested new directory."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"dynamics": validate_dynamics(), "bridge": benchmark_bridge()}
    (args.out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
