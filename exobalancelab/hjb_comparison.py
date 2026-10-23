"""Compare frozen-cohort HJB rollouts without dropping failures or changing targets.

Inputs are validation directories with protocol.json and outcomes.json. Outputs are
per-case CSV, aggregate JSON, and a PNG/PDF outcome matrix. All states use SI units.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(runs: dict[str, Path], output: Path) -> list[dict]:
    """Write paired outcomes; reject changed cohorts, plants, targets or missing cases."""
    if len(runs) < 2:
        raise ValueError("Provide at least two validation runs")
    reference = None
    rows, summaries, matrix = [], [], []
    for label, directory in runs.items():
        protocol_path = directory / "protocol.json"
        outcomes_path = directory / "outcomes.json"
        protocol = json.loads(protocol_path.read_text())
        outcomes = json.loads(outcomes_path.read_text())
        states = [(case["group"], case["state"]) for case in protocol["states"]]
        source = protocol["source"]
        signature = (
            states,
            source["profile"],
            source["source_model"],
            source["gravity_m_s2"],
            source["exo"],
            source["target"],
            source["duration_s"],
        )
        if reference is None:
            reference = signature
        if signature != reference:
            raise ValueError("Paired comparison requires identical states, plant, targets and horizon")
        if [(case["group"], case["state"]) for case in outcomes] != states:
            raise ValueError("Outcomes must contain every protocol state in its original order")
        statuses = []
        for index, case in enumerate(outcomes):
            if case["status"] == "rollout_error":
                category = "policy_error"
            elif case["status"] == "completed":
                category = "valid_target" if case["valid_target_reach"] else "completed_miss"
            else:
                raise ValueError(f"Unknown rollout status: {case['status']}")
            statuses.append(category)
            final = case.get("final_state", [None, None, None])
            rows.append(
                {
                    "run": label,
                    "case": index,
                    "original_group": case["group"],
                    "theta_rad": case["state"][0],
                    "velocity_rad_s": case["state"][1],
                    "human_torque_nm": case["state"][2],
                    "predicted_inside": case["predicted_inside"],
                    "value": case["value"],
                    "outcome": category,
                    "final_theta_rad": final[0],
                    "final_velocity_rad_s": final[1],
                    "final_human_torque_nm": final[2],
                    "error": case.get("error", ""),
                }
            )
        summaries.append(
            {
                "label": label,
                "source": source,
                "rollout": protocol.get("rollout"),
                "policy_sha256": protocol["policy_sha256"],
                "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "outcomes_sha256": hashlib.sha256(outcomes_path.read_bytes()).hexdigest(),
                "total": len(outcomes),
                "predicted_inside": sum(case["predicted_inside"] for case in outcomes),
                "predicted_inside_reached": sum(
                    case["predicted_inside"] and status == "valid_target"
                    for case, status in zip(outcomes, statuses, strict=True)
                ),
                **{key: statuses.count(key) for key in ["valid_target", "completed_miss", "policy_error"]},
            }
        )
        matrix.append(statuses)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (output / "cases.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    title = f"{source['profile']} / {source['exo']} assistance / {source['duration_s']:.3g} s"
    _plot(matrix, list(runs), [case[0] for case in reference[0]], title, output)
    return summaries


def _plot(matrix: list[list[str]], labels: list[str], groups: list[str], title: str, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    categories = ["valid_target", "completed_miss", "policy_error"]
    colors = ["#3b9986", "#e6b65a", "#9aa6b2"]
    names = ["Target reached + contact valid", "Completed; acceptance failed", "Rollout error"]
    values = np.array([[categories.index(status) for status in row] for row in matrix])
    figure, axis = plt.subplots(figsize=(10, 2.1 + 0.65 * len(labels)), layout="constrained")
    axis.imshow(values, cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5, aspect="auto")
    axis.set_xticks(range(len(groups)), [f"{i:02d}\n{group}" for i, group in enumerate(groups)])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xticks(np.arange(-0.5, len(groups), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=3)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_title("Same initial states and terminal targets · " + title, pad=16)
    axis.set_xlabel("Case and original selection group")
    axis.legend(
        handles=[Patch(facecolor=color, label=name) for color, name in zip(colors, names, strict=True)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    for extension in ["png", "pdf"]:
        figure.savefig(output / f"outcomes.{extension}", dpi=180)
    plt.close(figure)


def main() -> None:
    """Compare repeated --run LABEL=DIRECTORY arguments on one frozen cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=validation directory")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        label, separator, directory = item.partition("=")
        if not separator or not label or not directory or label in runs:
            parser.error("Each --run needs a unique LABEL=DIRECTORY")
        runs[label] = Path(directory)
    summaries = compare(runs, args.out)
    print(
        json.dumps(
            [
                {key: row[key] for key in ["label", "total", "valid_target", "policy_error"]}
                for row in summaries
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
