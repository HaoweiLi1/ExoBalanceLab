"""Freeze paired SI states, run native transport feedback in MuJoCo and assess refinement."""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from exobalancelab.paper import AUTHORS_PROFILES, exo_torque, ground_reaction
from exobalancelab.reachability import PolicyCoverageError
from exobalancelab.simulation import RunConfig, run
from exobalancelab.transport_policy import TransportPolicy

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def specification():
    result = json.loads((ROOT / "experiments/transport_feedback_v1.json").read_text())
    original = json.loads((ROOT / result["acceptance_source"]).read_text())["acceptance"]
    if any(value != original[key] for key, value in result["acceptance"].items()):
        raise ValueError("The original feedback acceptance limits must not change")
    return result


def matched_case(spec, profile, mode):
    return next(name for name in spec["families"] if name.startswith(f"{profile}_{mode}_"))


def select(policies, output, profile):
    spec = specification()
    path = output / f"selection_{profile}.json"
    if path.exists():
        return json.loads(path.read_text())
    names = [matched_case(spec, profile, mode) for mode in ["none", "gc"]]
    with (
        TransportPolicy(policies / names[0] / "policy.h5") as baseline,
        TransportPolicy(policies / names[1] / "policy.h5") as assisted,
    ):
        group, moving = baseline.goals[0]
        axes = tuple(axis[-1] for axis in moving)
        value = np.asarray(group["final_value"])
        for group, _ in baseline.goals[1:]:
            np.minimum(value, group["final_value"][:], out=value)
        random = np.random.default_rng(spec["seed"])
        near = np.argpartition(np.abs(value).ravel(), 1999)[:2000]
        # Sorting the small selected pool gives a deterministic order for tied values.
        near = near[np.lexsort((near, np.abs(value).ravel()[near]))]
        inside = np.flatnonzero(value < 0)
        pools = {"inside": random.permutation(inside), "boundary": random.permutation(near)}
        selected, seen = [], set()
        plant = AUTHORS_PROFILES[profile]
        for label in spec["groups"]:
            added = 0
            candidates = pools.get(label)
            if candidates is None:
                candidates = random.integers(0, value.size, size=200000)
            for index in candidates:
                index = int(index)
                if index in seen or (label == "outside" and value.flat[index] <= 0):
                    continue
                indices = np.unravel_index(index, value.shape)
                state = [float(axis[i]) for axis, i in zip(axes, indices, strict=True)]
                theta, speed, torque = state
                if speed < 0 or not plant.torque_min_nm <= torque <= plant.torque_max_nm:
                    continue
                q = theta - np.pi / 2
                reactions = [
                    ground_reaction(q, speed, torque + exo_torque(q, speed, plant, mode), plant)
                    for mode in ["none", "gc"]
                ]
                if any(min(r["normal_n"], r["cop_margin_m"], r["friction_margin_n"]) <= 0 for r in reactions):
                    continue
                try:
                    predictions = [min(policy.initial_values(state)) for policy in [baseline, assisted]]
                except PolicyCoverageError:
                    continue
                seen.add(index)
                selected.append(
                    dict(
                        index=len(selected),
                        group=label,
                        state=state,
                        baseline_native_index=index,
                        value=dict(zip(["none", "gc"], predictions, strict=True)),
                    )
                )
                added += 1
                if added == spec["per_group_per_population"]:
                    break
            if added != spec["per_group_per_population"]:
                raise ValueError("The predeclared selection pool could not fill " + label)
        result = dict(
            profile=profile,
            seed=spec["seed"],
            source_policy_sha256={names[0]: baseline.sha256, names[1]: assisted.sha256},
            samples=selected,
        )
        write(path, result)
        return result


def terminal_outcome(episode, target):
    metrics = episode.metrics
    final = np.array(
        [
            metrics["final_lean_rad"] + np.pi / 2,
            metrics["final_velocity_rad_s"],
            metrics["final_human_torque_nm"],
        ]
    )
    reached = any(
        np.all(
            abs(final - target[center])
            <= [target[f"{axis}{label}Range"] for axis in ["theta", "thetaDot", "torque"]]
        )
        for label, center in [("Ank", "ankleEq"), ("Toe", "toeEq")]
    )
    p = AUTHORS_PROFILES[episode.config.profile]
    reaction = ground_reaction(final[0] - np.pi / 2, final[1], final[2] + metrics["final_exo_torque_nm"], p)
    valid = (
        metrics["contact_valid_fraction"] == 1
        and min(reaction["normal_n"], reaction["cop_margin_m"], reaction["friction_margin_n"]) >= 0
    )
    return dict(
        final_state=final.tolist(),
        target_reached=bool(reached),
        contact_valid=bool(valid),
        valid_target_reach=bool(reached and valid),
    )


def execute(policies, output, profiles):
    spec = specification()
    frozen = output / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != spec:
        raise ValueError("Use a new output directory for a changed protocol")
    write(frozen, spec)
    sources = [Path(__file__), *sorted((ROOT / "exobalancelab").glob("*.py"))]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = output / "sources.json"
    if manifest.exists() and json.loads(manifest.read_text()) != hashes:
        raise ValueError("Use a new output directory for changed rollout sources")
    write(manifest, hashes)
    (output / "sources").mkdir(exist_ok=True)
    for path in sources:
        shutil.copy2(path, output / "sources" / path.name)
    selections = {profile: select(policies, output, profile) for profile in profiles}
    for profile in profiles:
        selection = selections[profile]
        for mode in ["none", "gc"]:
            name = matched_case(spec, profile, mode)
            source = json.loads((policies / name / "summary.json").read_text())
            for dt in spec["physics_steps_s"]:
                with TransportPolicy(policies / name / "policy.h5") as policy:
                    # Hash once before the controller's measured execution.
                    digest = policy.sha256
                    if digest != source["policy_sha256"] or digest != selection["source_policy_sha256"][name]:
                        raise ValueError("The rollout policy differs from export or frozen selection")
                    for case in selection["samples"]:
                        path = output / "episodes" / f"{profile}_{mode}_{case['index']:02d}_dt{dt:g}"
                        if (path / "result.json").exists():
                            continue
                        theta, speed, torque = case["state"]
                        config = RunConfig(
                            model="paper",
                            profile=profile,
                            parameters="authors",
                            exo=mode,
                            duration_s=spec["horizon_s"],
                            timestep_s=dt,
                            control_period_s=spec["control_period_s"],
                            command_delay_s=spec["command_delay_s"],
                            lean_rad=theta - np.pi / 2,
                            velocity_rad_s=speed,
                            initial_torque_nm=torque,
                        )
                        result = dict(
                            case,
                            profile=profile,
                            exo=mode,
                            config=asdict(config),
                            policy_sha256=digest,
                            predicted_inside=case["value"][mode] <= 0,
                        )
                        try:
                            episode = run(config, hjb=policy)
                        except (ValueError, RuntimeError) as error:
                            result.update(status="rollout_error", error=str(error))
                            if isinstance(error, PolicyCoverageError):
                                result.update(failed_query=error.query, policy_bounds=error.bounds)
                        else:
                            result.update(
                                status="completed",
                                metrics=episode.metrics,
                                **terminal_outcome(episode, source["target"]),
                            )
                            stride = round(0.001 / dt)
                            episode.arrays = {k: v[::stride] for k, v in episode.arrays.items()}
                            episode.save(path)
                            result["saved_interval_s"] = stride * dt
                        result["selected_goal"] = policy.selected_goal
                        write(path / "result.json", result)
                        print(path.name, result["status"], result.get("valid_target_reach"), flush=True)
    return report(output, profiles)


def report(output, profiles):
    spec = specification()
    if json.loads((output / "protocol.json").read_text()) != spec:
        raise ValueError("The recorded protocol changed")
    outcomes = []
    checks = []
    limits = np.array(
        [
            spec["acceptance"]["rollout_endpoint_abs"][name]
            for name in ["angle_rad", "velocity_rad_s", "torque_nm"]
        ]
    )
    for profile in profiles:
        selection = json.loads((output / f"selection_{profile}.json").read_text())
        for mode in ["none", "gc"]:
            for case in selection["samples"]:
                pair = [
                    json.loads(
                        (
                            output
                            / "episodes"
                            / f"{profile}_{mode}_{case['index']:02d}_dt{dt:g}"
                            / "result.json"
                        ).read_text()
                    )
                    for dt in spec["physics_steps_s"]
                ]
                outcomes.extend(pair)
                complete = all(p["status"] == "completed" for p in pair)
                stable = complete and all(
                    pair[0][k] == pair[1][k]
                    for k in ["target_reached", "contact_valid", "valid_target_reach"]
                )
                errors = abs(np.array(pair[0]["final_state"]) - pair[1]["final_state"]) if complete else None
                checks.append(
                    dict(
                        profile=profile,
                        exo=mode,
                        index=case["index"],
                        group=case["group"],
                        endpoint_abs=None if errors is None else errors.tolist(),
                        classification_stable=stable,
                        passed=bool(stable and np.all(errors <= limits)),
                    )
                )
    inside = [o for o in outcomes if o["predicted_inside"]]
    result = dict(
        profiles=profiles,
        episodes=len(outcomes),
        completed=sum(o["status"] == "completed" for o in outcomes),
        predicted_inside=len(inside),
        inside_valid_reach=sum(o.get("valid_target_reach", False) for o in inside),
        refinement=checks,
        refinement_passed=sum(c["passed"] for c in checks),
        refinement_total=len(checks),
        acceptance=spec["acceptance"],
        all_refinements_passed=all(c["passed"] for c in checks),
        deadline_passed=sum(
            o.get("metrics", {}).get("control_p99_us", float("inf"))
            <= spec["control_period_s"] * 1e6 * spec["acceptance"]["deadline_p99_fraction"]
            for o in outcomes
        ),
    )
    result["groups"] = []
    for profile in profiles:
        for mode in ["none", "gc"]:
            for label in spec["groups"]:
                rows = [
                    o
                    for o in outcomes
                    if o.get("profile") == profile
                    and o.get("exo") == mode
                    and o.get("group") == label
                    and o.get("config", {}).get("timestep_s") == spec["physics_steps_s"][-1]
                ]
                result["groups"].append(
                    dict(
                        profile=profile,
                        exo=mode,
                        group=label,
                        episodes=len(rows),
                        completed=sum(o["status"] == "completed" for o in rows),
                        predicted_inside=sum(o["predicted_inside"] for o in rows),
                        valid_reach=sum(o.get("valid_target_reach", False) for o in rows),
                    )
                )
    result["max_endpoint_abs"] = (
        np.max([c["endpoint_abs"] for c in checks if c["endpoint_abs"] is not None], axis=0).tolist()
        if any(c["endpoint_abs"] is not None for c in checks)
        else None
    )
    write(output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=["YF", "OM"], default=["OM", "YF"])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report(args.out, args.profiles)
    else:
        if args.policies is None:
            parser.error("--policies is required for new rollouts")
        execute(args.policies, args.out, args.profiles)
