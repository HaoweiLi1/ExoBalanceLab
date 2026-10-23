"""Refine every classification-unstable human triplet from the paired campaign.

This is a post-campaign diagnostic. It preserves the primary results, controller,
initial state and contact criterion, and retains all three assistance modes.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from validation_rollouts import execute, freeze, write

from exobalancelab.simulation import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[0.0000625, 0.00003125])
    args = parser.parse_args()
    source_path = args.input / "outcomes.json"
    records = json.loads(source_path.read_text())
    conditions = {}
    for case in records:
        config = case["config"]
        if config["model"] != "human" or config["timestep_s"] != 0.001:
            continue
        fine = next(
            r
            for r in records
            if r["config"]["timestep_s"] == 0.0005
            and all(r["config"][k] == config[k] for k in config if k != "timestep_s")
        )
        if case["status"] != fine["status"] or (
            case["status"] == "completed" and case["metrics"]["recovered"] != fine["metrics"]["recovered"]
        ):
            common = {k: v for k, v in config.items() if k not in ["exo", "timestep_s"]}
            conditions[json.dumps(common, sort_keys=True)] = common
    configs = [
        dict(condition, exo=mode, timestep_s=dt)
        for condition, dt, mode in itertools.product(
            conditions.values(), args.timesteps, ["none", "gc", "pd"]
        )
    ]
    specification = dict(
        study="post_campaign_contact_refinement",
        selection="Every human condition whose primary 1/0.5 ms recovery classification changes.",
        source_outcomes_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contact_criterion="Unchanged: positive support at every physics sample, with contact margins.",
        conditions=configs,
    )
    freeze(args.out, specification)
    outcomes, diagnostics = [], []
    for index, config in enumerate(configs):
        name = f"case{index:02d}_{config['exo']}_dt{config['timestep_s']:g}"
        path = args.out / "cases" / name
        result = execute(RunConfig(**config), path, {"id": name}, trace_interval=config["timestep_s"])
        outcomes.append(result)
        write(args.out / "outcomes.json", outcomes)
        if result["status"] != "completed":
            continue
        reference = next(
            r
            for r in records
            if r["config"]["timestep_s"] == 0.0005
            and all(r["config"][k] == config[k] for k in config if k != "timestep_s")
        )
        for key in ["initial_state", "feedback_gain", "feedback_reference"]:
            np.testing.assert_array_equal(result["metrics"][key], reference["metrics"][key])
        with np.load(path / "trace.npz") as trace:
            zero = trace["normal_n"] <= 0
            selected = np.flatnonzero(zero)
            window = (trace["time_s"] >= 0.09) & (trace["time_s"] <= 0.12)
            np.savez_compressed(path / "contact_window.npz", **{k: a[window] for k, a in trace.items()})
            diagnostics.append(
                dict(
                    id=name,
                    config=config,
                    recovered=result["metrics"]["recovered"],
                    zero_support_samples=int(zero.sum()),
                    zero_support_duration_s=float(zero.sum() * config["timestep_s"]),
                    first_zero_support_times_s=trace["time_s"][selected[:20]].tolist(),
                    minimum_normal_n=float(min(trace["normal_n"])),
                    fixed_initial_state_and_gains=True,
                )
            )
        write(args.out / "diagnosis.json", diagnostics)


if __name__ == "__main__":
    main()
