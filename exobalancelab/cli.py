"""Command-line experiments, MATLAB replay and paired MuJoCo videos; all units are SI."""

import argparse
import json
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from exobalancelab.bridge import MatlabController, Trajectory
from exobalancelab.paper import PROFILES
from exobalancelab.reachability import HJBPolicy
from exobalancelab.simulation import RunConfig, run


def main() -> None:
    """Run one experiment or a matched three-mode comparison from any working directory."""
    parser = argparse.ArgumentParser(description="ExoBalanceLab: MATLAB meets MuJoCo standing balance")
    parser.add_argument("command", choices=["run", "compare"])
    parser.add_argument("--model", choices=["paper", "contact", "human"], default="paper")
    parser.add_argument("--profile", choices=PROFILES, default="OF")
    parser.add_argument("--parameters", choices=["table_i", "authors"], default="table_i")
    parser.add_argument("--exo", choices=["none", "gc", "pd"], default="none")
    parser.add_argument("--duration", type=float, default=6)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--control-period", type=float, default=0.001)
    parser.add_argument("--lean", type=float, default=-0.03, help="q=theta-pi/2, rad; positive is forward")
    parser.add_argument("--velocity", type=float, default=0.15, help="rad/s; positive is forward")
    parser.add_argument("--push", type=float, default=0, help="anterior chest force N, human model only")
    parser.add_argument("--device-mass", type=float, default=2, help="total bilateral mass kg, human only")
    parser.add_argument("--strength", type=float, default=1, help="human MT and MRTD multiplier in (0,1]")
    parser.add_argument("--mt-scale", type=float, default=1, help="independent human torque multiplier")
    parser.add_argument("--mrtd-scale", type=float, default=1, help="independent torque-rate multiplier")
    parser.add_argument("--command-delay", type=float, default=0, help="sample-to-actuation delay, seconds")
    parser.add_argument("--human-feedback", choices=["default", "fixed", "readapted"], default="default")
    parser.add_argument("--initial-posture", choices=["root", "flat_feet"], default="root")
    parser.add_argument("--trajectory", type=Path, help="CSV or MAT desired human torque samples")
    parser.add_argument("--interpolation", choices=["linear", "zoh"], default="linear")
    parser.add_argument("--hjb-policy", type=Path, help="exported MATLAB/helperOC value function")
    parser.add_argument("--matlab-function", help="persistent Engine feedback function")
    parser.add_argument("--matlab-directory", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--out", type=Path, help="new output directory (default logs/<Singapore time>)")
    args = parser.parse_args()
    if args.matlab_function and (args.trajectory or args.matlab_directory is None):
        parser.error("MATLAB needs --matlab-directory and cannot be combined with --trajectory")
    if sum(bool(item) for item in [args.trajectory, args.matlab_function, args.hjb_policy]) > 1:
        parser.error("Choose one of trajectory, MATLAB feedback, or HJB policy")
    hjb = HJBPolicy(args.hjb_policy) if args.hjb_policy else None
    output = args.out or Path("logs") / datetime.now(ZoneInfo("Asia/Singapore")).strftime("%m%d_%H%M%S_%f")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    config = RunConfig(
        model=args.model,
        profile=args.profile,
        parameters=args.parameters,
        exo=args.exo,
        duration_s=args.duration,
        timestep_s=args.dt,
        control_period_s=args.control_period,
        lean_rad=args.lean,
        velocity_rad_s=args.velocity,
        device_mass_kg=args.device_mass,
        push_n=args.push,
        human_strength=args.strength,
        mt_scale=args.mt_scale,
        mrtd_scale=args.mrtd_scale,
        command_delay_s=args.command_delay,
        human_feedback=args.human_feedback,
        initial_posture=args.initial_posture,
    )
    config.validate()
    trajectory = Trajectory.load(args.trajectory, args.interpolation) if args.trajectory else None
    if hjb and args.command == "compare":
        parser.error("HJB policies are mode-specific; run each with its matching --exo")
    modes = ["none", "gc", "pd"] if args.command == "compare" else [args.exo]
    episodes = []
    for mode in modes:
        context = (
            MatlabController(args.matlab_function, args.matlab_directory, 6 if args.model == "human" else 1)
            if args.matlab_function
            else nullcontext(None)
        )
        with context as controller:
            episode = run(replace(config, exo=mode), trajectory, controller, hjb)
        episode.save(output / mode)
        episodes.append(episode)
        print(json.dumps(episode.metrics, indent=2))
    from exobalancelab.rendering import comparison_video, plot_comparison

    plot_comparison(episodes, output / "comparison.png")
    if args.video:
        comparison_video(episodes, output / "comparison.mp4")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
