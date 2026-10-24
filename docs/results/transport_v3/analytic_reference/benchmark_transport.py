"""Independent thin-target and state-constrained analytic transport benchmarks.

Both use x'=v, v'=u, |u|<=1, T=0.5 s and the unchanged goal |x|<=0.005,
|v|<=0.01. The constrained case also requires |v(t)|<=0.3 throughout.
Run in the JAX environment; rebuild reports from saved fields in the basic env.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from examples.hjb_analytic_reference import exact_bounds
from examples.report_validation import write

ROOT = Path(__file__).resolve().parents[1]


def constrained_bounds(velocity, duration=0.5, position_width=0.005, velocity_width=0.01, cap=0.3):
    """Integrate the exact upper/lower feasible velocity envelopes.

    The upper envelope min(v+t, cap, b+T-t) is achievable with bounded
    acceleration. Its breakpoints give the exact integral. Reflection gives
    the lower envelope. A separate control LP checks this formula in tests.
    """
    velocity = np.asarray(velocity, dtype=float)

    def maximum(v):
        times = np.stack(
            np.broadcast_arrays(
                np.zeros_like(v),
                np.full_like(v, duration),
                cap - v,
                np.full_like(v, duration + velocity_width - cap),
                (duration + velocity_width - v) / 2,
            )
        )
        times = np.sort(np.clip(times, 0, duration), axis=0)
        speeds = np.minimum(np.minimum(v[None, ...] + times, cap), velocity_width + duration - times)
        return np.sum(np.diff(times, axis=0) * (speeds[1:] + speeds[:-1]) / 2, axis=0)

    return (
        -position_width - maximum(velocity),
        position_width + maximum(-velocity),
        abs(velocity) <= min(cap, duration + velocity_width),
    )


def triple_bounds(velocity, duration=0.5, position_width=0.005, velocity_width=0.01, acceleration_width=0.02):
    """Exact position projection of the bounded-jerk reachable set.

    Backward flow maps the terminal box affinely. The support function's
    control polynomial s*(s/2-r) has at most one interior switch for normals
    (1,r,0). Integrating that bang-bang control gives the curved boundary;
    terminal acceleration/velocity limits contribute the two straight pieces.
    Independent control LPs verify the formula in tests.
    """
    speed = np.asarray(velocity, dtype=float)
    minimum = -velocity_width - duration * acceleration_width - duration**2 / 2

    def right(v):
        z = np.clip(v - minimum, 0, duration**2)
        corner = (
            position_width
            + duration * velocity_width
            + duration**2 * acceleration_width / 2
            + duration**3 / 6
        )
        curved = corner - z**1.5 / 3
        first = minimum + duration**2
        second = first + 2 * duration * acceleration_width
        return (
            curved
            - duration / 2 * np.clip(v - first, 0, 2 * duration * acceleration_width)
            - duration * np.maximum(v - second, 0)
        )

    return -right(-speed), right(speed), abs(speed) <= -minimum


def report(output):
    """Recompute every required absolute, spatial and temporal reference check."""
    manifest = json.loads((output / "manifest.json").read_text())
    specification = json.loads((ROOT / "experiments/transport_convergence_v3.json").read_text())
    limits = specification["analytic_acceptance"]
    if manifest["acceptance"] != limits:
        raise ValueError("The original analytic acceptance must not change")
    expected = sorted(
        (name, n, dt) for name, grids in specification["analytic_cases"].items() for n, dt in grids
    )
    actual = sorted((c["case"], c["resolution"], c["time_step_s"]) for c in manifest["cases"])
    if actual != expected:
        raise ValueError("Incomplete or different analytic cohort")
    for filename, digest in manifest["sources"].items():
        if hashlib.sha256((output / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Changed archived reference source: " + filename)
    rows, masks, exact_areas, temporal, spatial = [], {}, {}, [], []

    def iou(a, b):
        return float((a & b).sum() / (a | b).sum())

    for name, grids in specification["analytic_cases"].items():
        extents = (0.1, 0.3) if name == "triple" else (0.25, 0.75)
        x_axis, v_axis = [np.linspace(-bound, bound, 601) for bound in extents]
        x, v = np.meshgrid(x_axis, v_axis, indexing="ij")
        bound_function = {"free": exact_bounds, "constrained": constrained_bounds, "triple": triple_bounds}[
            name
        ]
        lower, upper, feasible = bound_function(v)
        exact = feasible & (x >= lower) & (x <= upper)

        def area(mask):
            return float(np.trapezoid(np.trapezoid(mask.astype(float), v_axis, axis=1), x_axis))

        exact_areas[name] = area(exact)
        for case in [c for c in manifest["cases"] if c["case"] == name]:
            path = output / case["field"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
                raise ValueError("Changed analytic field: " + str(path))
            with np.load(path) as data:
                value = RegularGridInterpolator(
                    (data["position"], data["velocity"]), data["value"], bounds_error=False, fill_value=1
                )(np.stack([x, v], -1))
            mask = value <= 0
            masks[name, case["resolution"], case["time_step_s"]] = mask
            row = dict(
                case,
                area=area(mask),
                area_error_percent=100 * abs(area(mask) - area(exact)) / area(exact),
                projection_iou=iou(mask, exact),
            )
            row["passed"] = (
                row["area_error_percent"] <= limits["area_relative_percent"]
                and row["projection_iou"] >= limits["projection_iou"]
            )
            rows.append(row)

        def compare(a, b, temporal_check):
            result = dict(
                case=name, area_error_percent=100 * abs(area(a) - area(b)) / area(b), projection_iou=iou(a, b)
            )
            result["passed"] = (
                result["area_error_percent"]
                <= limits["temporal_area_relative_percent" if temporal_check else "area_relative_percent"]
                and result["projection_iou"]
                >= limits["temporal_projection_iou" if temporal_check else "projection_iou"]
            )
            return result

        finest = max(n for n, dt in grids)
        coarse_dt, fine_dt = specification["temporal_comparison_steps_s"]
        result = compare(masks[name, finest, coarse_dt], masks[name, finest, fine_dt], True)
        temporal.append(dict(result, resolution=finest, coarse_step_s=coarse_dt, fine_step_s=fine_dt))
        if name == "triple":
            primary = specification["primary_time_step_s"]
            coarse_n, fine_n = sorted(n for n, dt in grids if dt == primary)[-2:]
            result = compare(masks[name, coarse_n, primary], masks[name, fine_n, primary], False)
            spatial.append(
                dict(result, coarse_resolution=coarse_n, fine_resolution=fine_n, time_step_s=primary)
            )
    result = dict(
        sources=manifest["sources"],
        acceptance=limits,
        exact_common_grid_areas=exact_areas,
        cases=rows,
        temporal=temporal,
        spatial=spatial,
        passed=all(r["passed"] for r in rows + temporal + spatial),
    )
    write(output / "summary.json", result)
    return result


def run(output, platform="gpu"):
    """Exercise two- and three-state propagation in the same numerical kernel."""
    import jax

    from exobalancelab import hjb_transport

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sources = hjb_transport.source_hashes()
    for filename in sources:
        (output / filename).write_bytes((ROOT / "exobalancelab" / filename).read_bytes())
    for filename in ["benchmark_transport.py", "hjb_analytic_reference.py"]:
        source = ROOT / "examples" / filename
        sources[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
        (output / filename).write_bytes(source.read_bytes())
    specification = json.loads((ROOT / "experiments/transport_convergence_v3.json").read_text())
    manifest = dict(sources=sources, acceptance=specification["analytic_acceptance"], cases=[])
    with jax.default_device(jax.devices(platform)[0]):
        for name, grids in specification["analytic_cases"].items():
            for n, dt in grids:
                started = time.perf_counter()
                if name == "triple":
                    axes, value = hjb_transport.triple_reference(n, dt)
                    position, velocity, field = axes[0], axes[1], value.min(axis=2)
                else:
                    axes, value = hjb_transport.reference(n, dt, np.inf if name == "free" else 0.3)
                    position, velocity, field = axes[1], axes[2], value[1]
                filename = f"{name}_n{n}_dt{dt}.npz"
                np.savez_compressed(output / filename, position=position, velocity=velocity, value=field)
                row = dict(
                    case=name,
                    resolution=n,
                    time_step_s=dt,
                    wall_s=time.perf_counter() - started,
                    field=filename,
                    sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                )
                manifest["cases"].append(row)
                write(output / "manifest.json", manifest)
                print(json.dumps(row), flush=True)
    return report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--platform", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    result = report(args.out) if args.report_only else run(args.out, args.platform)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
