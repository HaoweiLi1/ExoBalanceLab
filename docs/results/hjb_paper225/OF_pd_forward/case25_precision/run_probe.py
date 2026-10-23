"""Refine only the control/physics step for the recorded PD boundary target miss."""

import hashlib
import json
from pathlib import Path

import numpy as np

from exobalancelab.paper import AUTHORS_PROFILES, exo_torque, ground_reaction
from exobalancelab.reachability import HJBPolicy, PolicyCoverageError
from exobalancelab.simulation import RunConfig, run

source = Path("logs/hjb_paper225/validation/OF_pd_forward_25us")
output = Path("logs/hjb_paper225/validation/OF_pd_forward_case25_precision")
case = json.loads((source / "outcomes.json").read_text())[25]
summary = json.loads(Path("logs/hjb_paper225/solved/OF_pd_forward/summary.json").read_text())
steps = [0.0000125, 0.00000625]
protocol = {
    "source_case_index": 25,
    "source_protocol_sha256": hashlib.sha256((source / "protocol.json").read_bytes()).hexdigest(),
    "case": case,
    "steps_s": steps,
    "target": summary["target"],
    "scope": "One recorded target miss, unchanged policy/state/target; simulated control and physics step refinement.",
}
(output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
policy = HJBPolicy(
    "logs/hjb_paper225/solved/OF_pd_forward/policy.mat",
    gradient_cache="logs/hjb_gradient_cache/OF_pd_forward_225",
)
profile = AUTHORS_PROFILES["OF"]
rows = []
for step in steps:
    theta, speed, torque = case["state"]
    config = RunConfig(
        profile="OF", parameters="authors", exo="pd", duration_s=1.4,
        timestep_s=step, control_period_s=step, lean_rad=theta - np.pi / 2,
        velocity_rad_s=speed, initial_torque_nm=torque,
    )
    row = {"step_s": step}
    try:
        episode = run(config, hjb=policy)
    except (ValueError, RuntimeError) as error:
        row.update(status="rollout_error", error=str(error))
        if isinstance(error, PolicyCoverageError):
            row["failed_query"] = error.query
    else:
        episode.save(output / f"step_{step:.8f}")
        metrics = episode.metrics
        final = np.array([
            metrics["final_lean_rad"] + np.pi / 2,
            metrics["final_velocity_rad_s"], metrics["final_human_torque_nm"],
        ])
        target = summary["target"]
        reached = any(
            np.all(abs(final - target[center]) <= [target[f"theta{label}Range"],
                target[f"thetaDot{label}Range"], target[f"torque{label}Range"]])
            for label, center in [("Ank", "ankleEq"), ("Toe", "toeEq")]
        )
        lean = final[0] - np.pi / 2
        reactions = ground_reaction(
            lean, final[1], final[2] + exo_torque(lean, final[1], profile, "pd"), profile,
        )
        valid = metrics["contact_valid_fraction"] == 1 and min(
            reactions["normal_n"], reactions["cop_margin_m"], reactions["friction_margin_n"],
        ) >= 0
        row.update(status="completed", final_state=final.tolist(), target_reached=bool(reached),
            contact_valid=bool(valid), valid_target_reach=bool(reached and valid))
    rows.append(row)
    (output / "outcomes.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(row), flush=True)
