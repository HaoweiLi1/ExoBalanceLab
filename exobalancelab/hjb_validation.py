"""Compare a nominal MATLAB HJB prediction with matching MuJoCo forward rollouts.

Select inside/outside/boundary grid states before observing rollout results. Retain
the protocol, every outcome and successful episode records; errors are not recovery.
Coordinates are absolute theta [rad], speed [rad/s], and human torque [Nm].
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from exobalancelab.paper import AUTHORS_PROFILES, ground_reaction
from exobalancelab.reachability import HJBPolicy
from exobalancelab.simulation import RunConfig, run


def validate(run_dir: Path, output: Path, count: int = 3, seed: int = 240805418) -> dict:
    """Select and run count states per stratum; count must be positive."""
    if count < 1:
        raise ValueError("Count must be positive")
    source = json.loads((run_dir / "summary.json").read_text())
    profile = AUTHORS_PROFILES[source["profile"]]
    policy = HJBPolicy(run_dir / "policy.mat")
    policy.validate_plant(profile, "none")
    with h5py.File(policy.path) as f:
        axes = [np.asarray(f[key]).ravel() for key in ["theta_rad", "velocity_rad_s", "human_torque_nm"]]
        value = np.asarray(f["value"][-1]).T
    rng = np.random.default_rng(seed)
    pools = {
        "inside": np.flatnonzero(value < 0),
        "outside": np.flatnonzero(value > 0),
        "boundary": np.argsort(abs(value).ravel())[: min(2000, value.size)],
    }
    selected, seen = [], set()
    for group, pool in pools.items():
        added = 0
        for index in rng.permutation(pool):
            if int(index) in seen:
                continue
            indices = np.unravel_index(index, value.shape)
            theta, speed, torque = [float(axis[i]) for axis, i in zip(axes, indices, strict=True)]
            reaction = ground_reaction(theta - np.pi / 2, speed, torque, profile)
            if min(reaction["normal_n"], reaction["cop_margin_m"], reaction["friction_margin_n"]) <= 0:
                continue
            selected.append({"group": group, "state": [theta, speed, torque], "value": float(value[indices])})
            seen.add(int(index))
            added += 1
            if added == count:
                break
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "source": source,
        "policy_sha256": hashlib.sha256(policy.path.read_bytes()).hexdigest(),
        "seed": seed,
        "requested_per_group": count,
        "states": selected,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    outcomes = []
    for index, case in enumerate(selected):
        theta, speed, torque = case["state"]
        config = RunConfig(
            profile=profile.name,
            parameters="authors",
            duration_s=round(policy.horizon_s, 12),
            lean_rad=theta - np.pi / 2,
            velocity_rad_s=speed,
            initial_torque_nm=torque,
        )
        result = dict(case, predicted_inside=case["value"] <= 0)
        try:
            episode = run(config, hjb=policy)
        except (ValueError, RuntimeError) as error:
            result.update(status="rollout_error", error=str(error))
        else:
            episode.save(output / f"episode_{index:02d}")
            metrics = episode.metrics
            final = np.array(
                [
                    metrics["final_lean_rad"] + np.pi / 2,
                    metrics["final_velocity_rad_s"],
                    metrics["final_human_torque_nm"],
                ]
            )
            target = source["target"]
            reached = False
            for center, widths in [
                ("ankleEq", ["thetaAnkRange", "thetaDotAnkRange", "torqueAnkRange"]),
                ("toeEq", ["thetaToeRange", "thetaDotToeRange", "torqueToeRange"]),
            ]:
                reached |= bool(
                    np.all(abs(final - np.asarray(target[center])) <= [target[k] for k in widths])
                )
            reaction = ground_reaction(final[0] - np.pi / 2, final[1], final[2], profile)
            final_valid = (
                min(reaction["normal_n"], reaction["cop_margin_m"], reaction["friction_margin_n"]) >= 0
            )
            valid = metrics["contact_valid_fraction"] == 1 and final_valid
            result.update(
                status="completed",
                final_state=final.tolist(),
                target_reached=reached,
                contact_valid=bool(valid),
                valid_target_reach=bool(reached and valid),
            )
        outcomes.append(result)
        # Keep completed cases if a later episode or external interruption fails.
        (output / "outcomes.json").write_text(json.dumps(outcomes, indent=2, allow_nan=False) + "\n")
    import matplotlib.pyplot as plt

    projection = value.min(axis=2)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.pcolormesh(
        axes[0] - np.pi / 2, axes[1], (projection <= 0).T, shading="nearest", cmap="Blues", vmin=0, vmax=1
    )
    for group, marker in [("inside", "o"), ("outside", "x"), ("boundary", "+")]:
        points = np.array([case["state"] for case in selected if case["group"] == group])
        if len(points):
            axis.scatter(
                points[:, 0] - np.pi / 2, points[:, 1], marker=marker, label=group, color="darkorange"
            )
    axis.set(
        xlabel="Forward lean q (rad)",
        ylabel="Angular velocity (rad/s)",
        title=(
            f"{profile.name}: nominal HJB projection and preselected states\n"
            "Coarse grid; no convergence claim"
        ),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "projection.png", dpi=160)
    plt.close(figure)
    return {"selected": len(selected), "outcomes": outcomes}


def main() -> None:
    """Validate one solve_nominal_hjb output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=240805418)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_dir, args.out, args.count, args.seed), indent=2))


if __name__ == "__main__":
    main()
