"""Run the frozen timing and paired articulated-model experiments, keeping failures.

Only completed case files are reused. Full-rate metrics are calculated before traces
are decimated to 1 ms (HJB) or 10 ms (paired models) for compact inspection.
"""

import argparse
import hashlib
import itertools
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from exobalancelab.paper import AUTHORS_PROFILES, ground_reaction
from exobalancelab.reachability import HJBPolicy, PolicyCoverageError
from exobalancelab.simulation import RunConfig, run

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def freeze(output, specification):
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != specification:
            raise ValueError("Frozen protocol changed; use a new output directory")
    else:
        write(path, specification)
        write(
            output / "source_hashes.json",
            {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), *sorted((ROOT / "exobalancelab").glob("*.py"))]
            },
        )


def save_trace(episode, output, interval_s):
    output.mkdir(parents=True, exist_ok=True)
    stride = max(1, round(interval_s / episode.config.timestep_s))
    np.savez_compressed(output / "trace.npz", **{k: a[::stride] for k, a in episode.arrays.items()})
    (output / "model.xml").write_text(episode.xml)
    write(
        output / "trace_metadata.json",
        {"saved_interval_s": stride * episode.config.timestep_s, "metrics_use_all_physics_steps": True},
    )


def target_outcome(episode, source):
    metrics = episode.metrics
    final = np.array(
        [
            metrics["final_lean_rad"] + np.pi / 2,
            metrics["final_velocity_rad_s"],
            metrics["final_human_torque_nm"],
        ]
    )
    target = source["target"]
    reached = any(
        np.all(
            abs(final - target[center])
            <= [target[f"theta{label}Range"], target[f"thetaDot{label}Range"], target[f"torque{label}Range"]]
        )
        for center, label in [("ankleEq", "Ank"), ("toeEq", "Toe")]
    )
    p = AUTHORS_PROFILES[source["profile"]]
    q = final[0] - np.pi / 2
    grf = ground_reaction(q, final[1], final[2] + float(episode.arrays["exo_torque_nm"][-1]), p)
    contact = (
        metrics["contact_valid_fraction"] == 1
        and min(grf["normal_n"], grf["cop_margin_m"], grf["friction_margin_n"]) >= 0
    )
    return dict(
        final_state=final.tolist(),
        target_reached=bool(reached),
        contact_valid=bool(contact),
        valid_target_reach=bool(reached and contact),
    )


def execute(config, path, context, policy=None, source=None, trace_interval=0.01):
    record_path = path / "result.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    result = dict(context, config=asdict(config))
    started = time.perf_counter()
    try:
        episode = run(config, hjb=policy)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        result.update(status="rollout_error", error=str(error))
        if isinstance(error, PolicyCoverageError):
            result.update(
                failed_query=error.query,
                policy_bounds=error.bounds,
                elapsed_s=config.duration_s - error.query["remaining_s"],
            )
    else:
        result.update(status="completed", metrics=episode.metrics)
        if source is not None:
            result.update(target_outcome(episode, source))
        save_trace(episode, path, trace_interval)
    result["execution_wall_s"] = time.perf_counter() - started
    write(record_path, result)
    print(
        path.name
        + ": "
        + result["status"]
        + " "
        + str(result.get("valid_target_reach", result.get("metrics", {}).get("recovered"))),
        flush=True,
    )
    return result


def paired(protocol, output):
    p = protocol["paired_models"]
    configs = []
    for profile, model, feedback, dt in itertools.product(
        p["profiles"], p["models"], p["human_feedback"], [p["physics_dt_s"], p["physics_dt_s"] / 2]
    ):
        states = [(q, v, 0) for q, v in itertools.product(p["lean_rad"], p["velocity_rad_s"])]
        if model == "human":
            states += [(0, 0, force) for force in p["pushes_n"] if force != 0]
        for state_index, (q, v, push) in enumerate(states):
            pair = f"{profile}_{model}_{feedback}_dt{dt:g}_state{state_index:02d}"
            for mode in p["modes"]:
                config = RunConfig(
                    model=model,
                    profile=profile,
                    parameters="authors",
                    exo=mode,
                    human_feedback=feedback,
                    initial_posture="flat_feet" if model == "human" else "root",
                    duration_s=p["duration_s"],
                    timestep_s=dt,
                    control_period_s=p["control_period_s"],
                    lean_rad=q,
                    velocity_rad_s=v,
                    push_n=push,
                )
                configs.append(dict(id=pair + "_" + mode, pair=pair, config=asdict(config)))
    specification = {
        "study": "paired_models",
        "parent_protocol": protocol,
        "initialization_detail": (
            "Human ankles rotate with root lean; pelvis translation and velocity preserve settled sole "
            "positions and velocities. All initial states and fixed gains are checked across "
            "each assistance triplet."
        ),
        "cases": configs,
    }
    freeze(output, specification)
    results = []
    for item in configs:
        results.append(
            execute(
                RunConfig(**item["config"]),
                output / "cases" / item["id"],
                {"id": item["id"], "pair": item["pair"]},
            )
        )
    # Pair validation is an executed assertion, not just a protocol promise.
    for pair in sorted({r["pair"] for r in results}):
        group = [r for r in results if r["pair"] == pair and r["status"] == "completed"]
        if len(group) != 3:
            continue
        for result in group[1:]:
            np.testing.assert_array_equal(
                group[0]["metrics"]["initial_state"], result["metrics"]["initial_state"]
            )
            if result["config"]["human_feedback"] == "fixed":
                np.testing.assert_array_equal(
                    group[0]["metrics"]["feedback_gain"], result["metrics"]["feedback_gain"]
                )
    write(output / "outcomes.json", results)


def timing(protocol, output, policies_root=None):
    settings = protocol["timing"]
    selections = {
        ("none", "forward"): "OF_none_forward/admissible",
        ("none", "backward"): "OF_none_backward/admissible",
        ("gc", "forward"): "OF_gc_forward/50us",
        ("gc", "backward"): "OF_gc_backward/50us",
        ("pd", "forward"): "OF_pd_forward/25us",
        ("pd", "backward"): "OF_pd_backward/12p5us",
    }
    groups = []
    for mode, direction in itertools.product(settings["modes"], settings["directions"]):
        key = f"OF_{mode}_{direction}"
        selection_path = ROOT / "docs/results/hjb_paper225" / selections[mode, direction] / "protocol.json"
        selection = json.loads(selection_path.read_text())
        states = [next(s for s in selection["states"] if s["group"] == group) for group in settings["groups"]]
        source_path = ROOT / (
            "logs/hjb_gpu_OF_225_paper" if key == "OF_none_forward" else "logs/hjb_paper225/solved/" + key
        )
        if policies_root is not None:
            source_path = policies_root / key
        groups.append(
            dict(
                id=key,
                states=states,
                selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                policy_sha256=selection["policy_sha256"],
                source=str(source_path),
            )
        )
    freeze(output, {"study": "timing", "parent_protocol": protocol, "groups": groups})
    results = []
    for group in groups:
        source_dir = Path(group["source"])
        source = json.loads((source_dir / "summary.json").read_text())
        cache = output / "temporary_gradient" / group["id"]
        print("Loading policy and gradient " + group["id"], flush=True)
        policy = HJBPolicy(source_dir / "policy.mat", gradient_cache=cache)
        if policy.sha256 != group["policy_sha256"]:
            raise ValueError("Frozen selection and loaded policy have different hashes")
        conditions = [
            (settings["physics_dt_s"], period, delay)
            for period, delay in itertools.product(
                settings["control_periods_s"], settings["command_delays_s"]
            )
        ]
        conditions += [
            (settings["physics_reference_dt_s"], settings["physics_reference_control_period_s"], 0)
        ]
        for dt, period, delay in conditions:
            for state in group["states"]:
                name = f"{group['id']}_{state['group']}_dt{dt:g}_period{period:g}_delay{delay:g}"
                theta, velocity, torque = state["state"]
                config = RunConfig(
                    parameters="authors",
                    exo=source["exo"],
                    duration_s=1.4,
                    timestep_s=dt,
                    control_period_s=period,
                    command_delay_s=delay,
                    lean_rad=theta - np.pi / 2,
                    velocity_rad_s=velocity,
                    initial_torque_nm=torque,
                )
                results.append(
                    execute(
                        config,
                        output / "cases" / name,
                        dict(
                            id=name, policy_id=group["id"], state=state, predicted_inside=state["value"] <= 0
                        ),
                        policy=policy,
                        source=source,
                        trace_interval=0.001,
                    )
                )
        del policy
        # Only this study's generated, verified derivative cache is disposable.
        shutil.rmtree(cache)
        write(output / "outcomes.json", results)


def yf_validation(protocol, output, policies_root=None):
    """Broaden forward validation to the four newly solved YF baseline policies."""
    from exobalancelab.hjb_validation import validate

    specification = {
        "study": "YF_forward_validation",
        "seed": protocol["seed"],
        "count_per_group": 3,
        "physics_dt_s": 0.0000125,
        "control_period_s": 0.001,
        "command_delay_s": 0,
        "cases": [
            f"YF_{mode}_{direction}_source_entry_mt100_mrtd100_n151_cfl0.8"
            for mode, direction in itertools.product(["none", "gc"], ["forward", "backward"])
        ],
        "scope": "Preselected inside/boundary/outside states in a second population; no cohort replacement.",
    }
    freeze(output, specification)
    for case in specification["cases"]:
        source = (policies_root or ROOT / "logs/validation_v1/hjb/solved") / case
        cache = output / "temporary_gradient" / case
        destination = output / "cases" / case
        if destination.exists():
            raise FileExistsError("Use a new directory for YF forward validation")
        print("Validating " + case, flush=True)
        result = validate(
            source,
            destination,
            count=3,
            seed=protocol["seed"],
            timestep_s=0.0000125,
            control_period_s=0.001,
            gradient_cache=cache,
        )
        if result["selected"] != 9:
            raise ValueError("YF validation did not fill all three frozen sampling groups")
        shutil.rmtree(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["paired", "timing", "yf"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/validation_v1.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--policies-root", type=Path, help="Directory of complete policy case folders for timing or YF"
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.study == "paired":
        if args.policies_root is not None:
            parser.error("Paired models do not use HJB policies")
        paired(protocol, args.out.resolve())
    else:
        {"timing": timing, "yf": yf_validation}[args.study](
            protocol, args.out.resolve(), args.policies_root.resolve() if args.policies_root else None
        )


if __name__ == "__main__":
    main()
