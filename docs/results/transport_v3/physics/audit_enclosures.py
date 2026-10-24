"""Check the exact enclosure-axis updates without allocating value fields."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np

from exobalancelab.hjb_transport import (
    SPLIT_STAGES,
    angle,
    angle_axis,
    control,
    control_axis,
    paper_problem,
    source_hashes,
    velocity,
    velocity_axis,
)


@jax.jit
def expand(axes, problem, dt, count):
    updates = {control: (2, control_axis), angle: (0, angle_axis), velocity: (1, velocity_axis)}

    def step(_, axes):
        for operator, fraction in SPLIT_STAGES:
            dimension, function = updates[operator]
            destination = function(axes, problem, dt * fraction)
            axes = tuple(destination if i == dimension else a for i, a in enumerate(axes))
        return axes

    return jax.lax.fori_loop(0, count, step, axes)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
rows = []
for path in sorted(args.inputs.glob("*.json")):
    inputs = json.loads(path.read_text())
    lower, upper = np.array(inputs["lower"]), np.array(inputs["upper"])
    summary = inputs["summary"]
    problem = paper_problem(summary, lower, upper)
    boxes = []
    for label, key in [("Ank", "ankleEq"), ("Toe", "toeEq")]:
        center = np.array(summary["target"][key])
        widths = np.array([summary["target"][a + label + "Range"] for a in ["theta", "thetaDot", "torque"]])
        if any(np.array_equal(center, c) and np.array_equal(widths, w) for c, w in boxes):
            continue
        boxes.append((center, widths))
        for requested in [0.01, 0.005]:
            lo, hi = np.maximum(lower, center - 2 * widths), np.minimum(upper, center + 2 * widths)
            assert np.all(lo < hi)
            # Only axis endpoints enter these expansion bounds. Two entries retain
            # the exact maintained axis operators without a dense value function.
            axes = tuple(j.array([a, b]) for a, b in zip(lo, hi))
            count = math.ceil(summary["duration_s"] / requested)
            final = np.array(expand(axes, problem, summary["duration_s"] / count, count))
            error = float(np.max(abs(final - np.stack([lower, upper], axis=1))))
            rows.append(
                dict(
                    family=path.stem,
                    goal=label,
                    time_step_s=requested,
                    max_endpoint_error=error,
                    passed=error <= 1e-12,
                )
            )
assert len(rows) == 68
result = dict(
    sources=source_hashes(),
    audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope="Computational enclosure endpoints only; no value-field convergence claim",
    cases=rows,
    passed=all(r["passed"] for r in rows),
)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        dict(
            checks=len(rows),
            passed=result["passed"],
            maximum_endpoint_error=max(r["max_endpoint_error"] for r in rows),
        ),
        indent=2,
    )
)
