"""Report the fixed OM target intervention in rad²/s and percentage points.

Read one completed diagnostic and its two retained 225-grid reference runs.
Use the same signed-velocity area calculation as the population comparison.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from exobalancelab.hjb_regions import _case_data, _require_solved


def report(protocol_path: Path, output: Path) -> dict:
    """Compare computed and source-entry targets on the same reconciled plant."""
    protocol = json.loads(protocol_path.read_text())
    candidate = protocol["cases"][0]
    if len(protocol["cases"]) != 1 or (candidate["profile"], candidate["exo"], candidate["direction"]) != (
        "OM",
        "gc",
        "forward",
    ):
        raise ValueError("This diagnostic requires one OM/GC/forward case")
    cases = [
        dict(candidate, exo="none", solved=protocol["reference_nominal_solved"]),
        dict(candidate, solved=protocol["reference_gc_solved"]),
        candidate,
    ]
    _require_solved(cases)
    loaded = [_case_data(case, protocol) for case in cases]
    old, new = loaded[1][1], loaded[2][1]
    for key in ["solver_sha256", "precision", "versions"]:
        if any(item[1][key] != loaded[0][1][key] for item in loaded[1:]):
            raise ValueError(f"Different {key} in the retained comparison")
    if any(not np.array_equal(a, b) for a, b in zip(loaded[1][2][:2], loaded[2][2][:2], strict=True)):
        raise ValueError("The target intervention changed projection axes")
    changed = {key for key in old["target"] if old["target"][key] != new["target"][key]}
    if set(old["target"]) != set(new["target"]) or changed != {"thetaDotAnkRange", "torqueAnkRange"}:
        raise ValueError("The recorded target intervention changed other target parameters")
    if new["target"]["thetaDotAnkRange"] != 0.01 or new["target"]["torqueAnkRange"] != 1.0:
        raise ValueError("Candidate does not use the released entry's fixed bounds")
    labels = ["No assistance", "GC / computed target", "GC / source-entry ankle bounds"]
    rows = [dict(variant=label, **item[0]) for label, item in zip(labels, loaded, strict=True)]
    nominal, computed, overridden = [row["area_rad2_per_s"] for row in rows]
    if nominal <= 0 or computed <= 0:
        raise ValueError("The recorded positive-area references are required for percentages")
    computed_effect = 100 * (computed / nominal - 1)
    overridden_effect = 100 * (overridden / nominal - 1)
    paper = protocol["trigger"]["paper_change_percent"]
    result = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "area_method": "Trapezoidal integral of sampled min_torque(V)<=0 at nonnegative velocity.",
        "areas": rows,
        "metrics": {
            "computed_target_gc_change_percent": computed_effect,
            "source_bounds_gc_change_percent": overridden_effect,
            "target_intervention_area_change_percent": 100 * (overridden / computed - 1),
            "target_intervention_effect_percentage_points": overridden_effect - computed_effect,
            "paper_gc_change_percent": paper,
            "computed_target_minus_paper_percentage_points": computed_effect - paper,
            "source_bounds_minus_paper_percentage_points": overridden_effect - paper,
        },
        "scope": protocol["interpretation"],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "areas.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = report(arguments.protocol, arguments.output)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
