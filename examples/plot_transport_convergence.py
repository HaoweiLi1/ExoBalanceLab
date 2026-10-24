"""Plot the recorded spatial checks without rerunning numerical integration."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from examples.benchmark_transport import constrained_bounds, report, triple_bounds
from examples.hjb_analytic_reference import exact_bounds


def label(family):
    population, assistance, direction, *rest = family.split("_")
    target = " ".join(rest[:-2])
    strength = "MT50" if rest[-2] == "mt50" else "MRTD50" if rest[-1] == "mrtd50" else "nominal"
    detail = target if population == "OM" else strength
    short_direction = {"forward": "fwd", "backward": "back"}[direction]
    return f"{population} · {assistance.upper()} · {short_direction} · {detail}"


def plot(source, output):
    report = json.loads(source.read_text())
    checks = report["spatial"][-report["spatial_total"] :]
    limits = report["acceptance"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 7.6), sharey=True)
    colors = ["#267a9f" if check["passed"] else "#c15737" for check in checks]
    position = np.arange(len(checks))
    columns = [
        ("area_error_percent", "Area difference (%)", limits["spatial_area_relative_percent"]),
        ("assistance_effect_change_pp", "GC effect change (pp)", limits["spatial_assistance_effect_pp"]),
        ("projection_iou", "Projected-set IoU", limits["common_grid_projection_iou"]),
    ]
    for axis, (key, title, threshold) in zip(axes, columns):
        values = [check[key] for check in checks]
        axis.scatter(values, position, c=colors, s=32, zorder=3)
        axis.axvline(threshold, color="#c15737", linestyle="--", linewidth=1.2)
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        if key == "projection_iou":
            axis.set_xlim(min(min(values) - 0.002, threshold - 0.005), 1.001)
        else:
            axis.set_xlim(-0.04 * threshold, max(max(values), threshold) * 1.18)
    axes[0].set_yticks(position, [label(check["family"]) for check in checks], fontsize=9)
    axes[0].invert_yaxis()
    coarse, fine = report["finest_spatial_pair"]
    figure.suptitle(f"Frozen human-model families · {coarse}³ → {fine}³", fontsize=15, y=0.98)
    figure.text(
        0.51,
        0.925,
        "Dashed lines: unchanged acceptance limits   ·   Blue: all three checks pass",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def plot_reference(source, output):
    measured = report(source)
    continuous_checks = []
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 4.5))
    definitions = [
        ("free", "Double integrator", exact_bounds, 0.15, 0.55),
        ("constrained", "Velocity-constrained", constrained_bounds, 0.15, 0.33),
        ("triple", "Three-state bounded jerk", triple_bounds, 0.045, 0.16),
    ]
    for axis, (name, title, bounds, x_limit, v_limit) in zip(axes, definitions):
        candidates = [c for c in measured["cases"] if c["case"] == name]
        chosen = max(candidates, key=lambda c: (c["resolution"], -c["time_step_s"]))
        feasible_limit = {"free": 0.51, "constrained": 0.3, "triple": 0.145}[name]

        def width(speed):
            lower, upper, feasible = bounds(speed)
            return float(upper - lower) if feasible else 0.0

        exact_area, error_estimate = quad(
            width, -feasible_limit, feasible_limit, points=[-0.01, 0, 0.01], epsabs=1e-12, epsrel=1e-12
        )
        continuous_checks.append(
            dict(
                case=name,
                continuous_exact_area=exact_area,
                quadrature_error_estimate=error_estimate,
                maximum_area_error_percent=max(100 * abs(c["area"] / exact_area - 1) for c in candidates),
            )
        )
        speed = np.linspace(-v_limit, v_limit, 3001)
        left, right, feasible = bounds(speed)
        axis.fill_betweenx(speed[feasible], left[feasible], right[feasible], color="#e5eff4")
        axis.plot(left[feasible], speed[feasible], color="#333333", linestyle="--", linewidth=1)
        axis.plot(right[feasible], speed[feasible], color="#333333", linestyle="--", linewidth=1)
        with np.load(source / chosen["field"]) as data:
            axis.contour(data["position"], data["velocity"], data["value"].T, levels=[0], colors=["#267a9f"])
        axis.set_title(
            f"{title}\nArea error {chosen['area_error_percent']:.3f}% · IoU {chosen['projection_iou']:.5f}",
            fontsize=11,
        )
        axis.set(xlim=(-x_limit, x_limit), ylim=(-v_limit, v_limit), xlabel="Position x", ylabel="Velocity v")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    figure.suptitle("Independent analytic checks · 0.5 s horizon", fontsize=15)
    figure.text(
        0.5,
        0.025,
        "Dashed: exact boundary   ·   Blue: numerical zero contour at 5 ms",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.93), w_pad=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    output.with_suffix(".json").write_text(
        json.dumps(
            dict(reference_sources=measured["sources"], continuous_area_checks=continuous_checks), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--report", type=Path)
    selection.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.reference:
        plot_reference(args.reference, args.out)
    else:
        plot(args.report, args.out)
