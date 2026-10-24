"""Compare complete HJB histories on identical physical grids and target sets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def compare(reference: Path, candidate: Path, output: Path) -> dict:
    summaries = [json.loads((path / "summary.json").read_text()) for path in [reference, candidate]]
    if any(summary.get("stage", "solved") != "solved" for summary in summaries):
        raise ValueError("Cannot compare an unfinished solve")
    for key in ["profile", "direction", "method"]:
        if summaries[0][key] != summaries[1][key]:
            raise ValueError(f"Different HJB {key}")
    if summaries[0].get("exo", "none") != summaries[1].get("exo", "none"):
        raise ValueError("Different assistance")
    for group in ["target", "source_model"]:
        if set(summaries[0][group]) != set(summaries[1][group]):
            raise ValueError(f"Different {group} fields")
        for key in summaries[0][group]:
            if not np.allclose(summaries[0][group][key], summaries[1][group][key], atol=1e-12, rtol=1e-12):
                raise ValueError(f"Different {group}.{key}")
    records = []
    with h5py.File(reference / "policy.mat") as original, h5py.File(candidate / "policy.mat") as new:
        contracts = [
            json.loads("".join(chr(int(c)) for c in file["contract_json"][:].ravel()))
            for file in [original, new]
        ]
        if contracts[0] != contracts[1]:
            raise ValueError("Different physical policy contracts")
        axes = []
        for key in ["theta_rad", "velocity_rad_s", "human_torque_nm", "time_to_go_s"]:
            axis = np.asarray(original[key]).ravel()
            other = np.asarray(new[key]).ravel()
            if axis.shape != other.shape or not np.allclose(axis, other, atol=1e-12, rtol=0):
                raise ValueError(f"Different {key}")
            axes.append(axis)
        if original["value"].shape != new["value"].shape:
            raise ValueError("Different history shape")
        if not np.array_equal(original["value"][0], new["value"][0]):
            raise ValueError("Different initial target/constraint field")
        for index, time in enumerate(axes[3]):
            baseline, actual = [np.asarray(file["value"][index]).T for file in [original, new]]
            if not (np.isfinite(baseline).all() and np.isfinite(actual).all()):
                raise ValueError("Nonfinite value field")
            delta = actual - baseline
            negative = baseline <= 0
            predicted = actual <= 0
            union = np.count_nonzero(negative | predicted)
            records.append(
                {
                    "time_s": float(time),
                    "max_absolute_error": float(np.max(abs(delta))),
                    "rmse": float(np.sqrt(np.mean(delta**2))),
                    "sign_disagreement_nodes": int(np.count_nonzero(negative != predicted)),
                    "negative_set_jaccard": float(np.count_nonzero(negative & predicted) / union)
                    if union
                    else 1.0,
                    "reference_nonpositive_nodes": int(np.count_nonzero(negative)),
                    "candidate_nonpositive_nodes": int(np.count_nonzero(predicted)),
                }
            )
    projected = [field.min(axis=2) for field in [baseline, actual]]
    velocity = axes[1]
    direction = summaries[0]["direction"]
    selected = velocity >= 0 if direction == "forward" else velocity <= 0
    areas = [
        float(
            np.trapezoid(
                np.trapezoid((field[:, selected] <= 0).astype(float), velocity[selected], axis=1), axes[0]
            )
        )
        for field in projected
    ]
    report = {
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_wall_s": summaries[0]["wall_s"],
        "candidate_wall_s": summaries[1]["wall_s"],
        "final": records[-1],
        "area_rad2_per_s": dict(zip(["reference", "candidate"], areas, strict=True)),
        "relative_area_difference": (areas[1] - areas[0]) / areas[0] if areas[0] else None,
        "projected_sign_disagreement_nodes": int(
            np.count_nonzero((projected[0] <= 0) != (projected[1] <= 0))
        ),
        "area_method": "Trapezoidal integral of sampled min_torque(V)<=0 over the selected velocity sign.",
        "scope": "Backend comparison on one grid; no spatial convergence or bitwise-equivalence claim.",
    }
    report["policy_sha256"] = {}
    for name, directory in [("reference", reference), ("candidate", candidate)]:
        with (directory / "policy.mat").open("rb") as file:
            report["policy_sha256"][name] = hashlib.file_digest(file, "sha256").hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for field, color, label in zip(projected, ["#205b9e", "#d05a24"], ["MATLAB", "JAX"], strict=True):
        axis.contour(axes[0], axes[1], field.T, levels=[0], colors=[color], linewidths=1.5)
        axis.plot([], [], color=color, label=label)
    axis.set(
        xlabel="Absolute angle (rad)",
        ylabel="Angular velocity (rad/s)",
        title=f"{summaries[0]['profile']} / {summaries[0].get('exo', 'none')}: same-grid backend comparison",
    )
    axis.legend()
    figure.tight_layout()
    for extension in ["png", "pdf"]:
        figure.savefig(output / f"projection.{extension}", dpi=160)
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.candidate, args.out), indent=2))


if __name__ == "__main__":
    main()
