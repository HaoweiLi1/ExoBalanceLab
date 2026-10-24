"""Predeclared controller-specific recovery grid, distinct from an HJB reachable set."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from exobalancelab.simulation import RunConfig, run


def main() -> None:
    """Run every grid point and assistance mode, keeping failures and the protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--duration", type=float, default=6)
    parser.add_argument("--profile", default="OF")
    args = parser.parse_args()
    if args.points < 2:
        parser.error("Use at least two grid points on each axis")
    args.out.mkdir(parents=True, exist_ok=False)
    angles = np.linspace(-0.08, 0.2, args.points)
    speeds = np.linspace(-0.6, 0.6, args.points)
    config = RunConfig(
        profile=args.profile, duration_s=args.duration, timestep_s=0.002, control_period_s=0.002
    )
    config.validate()
    protocol = {
        "config": asdict(config),
        "angles_rad": angles.tolist(),
        "velocities_rad_s": speeds.tolist(),
        "human_initial_torque_nm": 0,
        "modes": ["none", "gc", "pd"],
        "success": "last 1 s |q|<.05 rad and |qdot|<.05 rad/s; C3-C5 valid at all steps",
        "scope": "fixed adaptive-PD recovery slice, not maximal HJB stabilizable region",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = []
    with (args.out / "grid.csv").open("w") as file:
        writer = None
        for qi, q in enumerate(angles):
            for velocity in speeds:
                for mode in protocol["modes"]:
                    config.lean_rad, config.velocity_rad_s, config.exo = float(q), float(velocity), mode
                    episode = run(config)
                    row = {
                        "lean_rad": q,
                        "velocity_rad_s": velocity,
                        "exo": mode,
                        "recovered": int(episode.metrics["recovered"]),
                        "contact_valid_fraction": episode.metrics["contact_valid_fraction"],
                        "max_abs_lean_rad": episode.metrics["max_abs_lean_rad"],
                    }
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=list(row), lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(row)
                    results.append(row)
            file.flush()
            print(f"Completed angle {qi + 1}/{len(angles)}", flush=True)
    recovery = {
        mode: np.array([r["recovered"] for r in results if r["exo"] == mode]).reshape(
            args.points, args.points
        )
        for mode in protocol["modes"]
    }
    summary = {}
    for mode in protocol["modes"]:
        summary[mode] = {
            "recovered_count": int(recovery[mode].sum()),
            "total": args.points**2,
            "helped_vs_none": int(np.sum((recovery[mode] == 1) & (recovery["none"] == 0))),
            "hindered_vs_none": int(np.sum((recovery[mode] == 0) & (recovery["none"] == 1))),
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, protocol["modes"]):
        # 0: neither, 1: loses recovery, 2: gains recovery, 3: both recover.
        code = recovery["none"] + 2 * recovery[mode]
        ax.imshow(
            code.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=3,
            extent=[angles[0], angles[-1], speeds[0], speeds[-1]],
            cmap=ListedColormap(["#e5e9ee", "#e67b67", "#63bda3", "#5586ac"]),
        )
        ax.set_title(f"{mode.upper()} | recovered {summary[mode]['recovered_count']}/{args.points**2}")
        ax.set_xlabel("q = theta - pi/2 (rad)")
    axes[0].set_ylabel("Angular velocity (rad/s)")
    figure.suptitle(
        "Fixed-controller slice | blue: both recover | green: helped | red: hindered\n"
        "Zero initial human torque; not the paper's HJB stabilizable region",
        fontsize=11,
    )
    figure.savefig(args.out / "recovery.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
