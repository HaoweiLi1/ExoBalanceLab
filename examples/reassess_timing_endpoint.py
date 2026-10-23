"""Reassess early timing results using recorded held device torque at the endpoint.

Retain original outcome files and classification fields. This corrects analysis only;
no trajectory, selected state, control command or measured latency is changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from exobalancelab.paper import AUTHORS_PROFILES, ground_reaction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path, required=True)
    args = parser.parse_args()
    path = args.timing / "outcomes.json"
    original = args.timing / "outcomes_instantaneous_endpoint.json"
    if original.exists():
        raise FileExistsError("Reassessment already recorded")
    raw = path.read_bytes()
    results = json.loads(raw)
    changed = []
    for r in results:
        if r["status"] != "completed":
            continue
        c = r["config"]
        with np.load(args.timing / "cases" / r["id"] / "trace.npz") as trace:
            # Check that decimation retained a sample after the final delivery.
            last_tick = np.floor(
                (c["duration_s"] - c["timestep_s"] - c["command_delay_s"]) / c["control_period_s"] + 1e-10
            )
            last_arrival = last_tick * c["control_period_s"] + c["command_delay_s"]
            if trace["time_s"][-1] < last_arrival - 1e-10:
                raise ValueError("Saved trace cannot identify the final held command")
            torque = float(trace["exo_torque_nm"][-1])
        final = r["final_state"]
        reaction = ground_reaction(
            final[0] - np.pi / 2, final[1], final[2] + torque, AUTHORS_PROFILES[c["profile"]]
        )
        valid = (
            r["metrics"]["contact_valid_fraction"] == 1
            and min(reaction["normal_n"], reaction["cop_margin_m"], reaction["friction_margin_n"]) >= 0
        )
        r["original_contact_valid"] = r["contact_valid"]
        r["original_valid_target_reach"] = r["valid_target_reach"]
        r["contact_valid"] = bool(valid)
        r["valid_target_reach"] = bool(valid and r["target_reached"])
        r["metrics"]["final_exo_torque_nm"] = torque
        if r["original_valid_target_reach"] != r["valid_target_reach"]:
            changed.append(r["id"])
    original.write_bytes(raw)
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    for r in results:
        (args.timing / "cases" / r["id"] / "result.json").write_text(
            json.dumps(r, indent=2, allow_nan=False) + "\n"
        )
    report = dict(
        reason=__doc__,
        original_outcomes_sha256=hashlib.sha256(raw).hexdigest(),
        corrected_outcomes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        changed_classifications=changed,
        assessment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.timing / "endpoint_reassessment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
