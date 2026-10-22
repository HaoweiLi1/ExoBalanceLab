"""Aggregate a complete five-profile HJB campaign and compare paper Table II."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from exobalancelab.paper import AUTHORS_PROFILES

PROFILES = ("YM", "YF", "OM", "OF", "wOF")
MODES = ("none", "gc", "pd")
DIRECTIONS = ("forward", "backward")


def report(protocol_path: Path, output: Path) -> dict:
    """Require all 30 solves; retain zero-area cases without inventing percentages."""
    protocol = json.loads(protocol_path.read_text())
    if protocol["parameter_set"] != "authors":
        raise ValueError("This campaign requires the pinned author parameter set")
    expected = {(p, m, d) for p in PROFILES for m in MODES for d in DIRECTIONS}
    cases = protocol["cases"]
    keys = [(c["profile"], c["exo"], c["direction"]) for c in cases]
    if len(keys) != 30 or set(keys) != expected:
        raise ValueError(
            "Table II comparison requires exactly all five profiles, three modes and two directions"
        )
    missing = []
    for case in cases:
        path = Path(case["solved"])
        if not (path / "summary.json").exists():
            missing.append(str(path))
        elif json.loads((path / "summary.json").read_text()).get("stage") != "solved":
            missing.append(str(path))
    if missing:
        raise ValueError("Incomplete HJB campaign: " + ", ".join(missing))
    rows = []
    for case in cases:
        path = Path(case["solved"])
        summary = json.loads((path / "summary.json").read_text())
        profile = AUTHORS_PROFILES[case["profile"]]
        expected_model = {
            "m": profile.mass_kg,
            "h": profile.height_m,
            "tauMin": profile.torque_min_nm,
            "tauMax": profile.torque_max_nm,
            "uMin": profile.rate_min_nm_s,
            "uMax": profile.rate_max_nm_s,
        }
        if summary["gravity_m_s2"] != profile.gravity_m_s2 or any(
            not np.isclose(summary["source_model"][key], value, rtol=1e-12, atol=1e-12)
            for key, value in expected_model.items()
        ):
            raise ValueError(f"Different physical model in {path}")
        for key in ["profile", "exo", "direction"]:
            if summary[key] != case[key]:
                raise ValueError(f"Mismatched {key} in {path}")
        for key, expected_value in [
            ("duration_s", protocol["horizon_s"]),
            ("output_step_s", protocol["output_step_s"]),
            ("velocity_buffer_rad_s", protocol["velocity_buffer_rad_s"]),
            ("requested_resolution", protocol["requested_resolution"]),
        ]:
            if not np.allclose(summary[key], expected_value, rtol=0, atol=1e-12):
                raise ValueError(f"Mismatched {key} in {path}")
        if summary["method"] != protocol["method"] or summary["aligned_zero"] != protocol["align_zero"]:
            raise ValueError(f"Mismatched method or grid alignment in {path}")
        projection = path / "projection.mat"
        arrays = loadmat(projection)
        theta = arrays["theta_rad"].ravel()
        velocity = arrays["velocity_rad_s"].ravel()
        field = arrays["projected_value"]
        if (
            field.shape != (len(theta), len(velocity))
            or not np.isfinite(field).all()
            or not np.isfinite(theta).all()
            or not np.isfinite(velocity).all()
            or np.any(np.diff(theta) <= 0)
            or np.any(np.diff(velocity) <= 0)
        ):
            raise ValueError(f"Invalid projected value grid in {path}")
        selected = velocity >= 0 if case["direction"] == "forward" else velocity <= 0
        area = float(
            np.trapezoid(
                np.trapezoid((field[:, selected] <= 0).astype(float), velocity[selected], axis=1), theta
            )
        )
        rows.append(
            {
                "profile": case["profile"],
                "exo": case["exo"],
                "direction": case["direction"],
                "area_rad2_per_s": area,
                "theta_min_rad": float(theta[0]),
                "theta_max_rad": float(theta[-1]),
                "velocity_min_rad_s": float(velocity[0]),
                "velocity_max_rad_s": float(velocity[-1]),
                "resolution": "x".join(map(str, summary["resolution"])),
                "wall_s": summary["wall_s"],
                "run": str(path),
                "projection_sha256": hashlib.sha256(projection.read_bytes()).hexdigest(),
            }
        )
    lookup = {(r["profile"], r["exo"], r["direction"]): r for r in rows}
    comparisons = []
    for profile in PROFILES:
        for exo in ["gc", "pd"]:
            for direction in DIRECTIONS:
                baseline = lookup[profile, "none", direction]["area_rad2_per_s"]
                assisted = lookup[profile, exo, direction]["area_rad2_per_s"]
                observed = 100 * (assisted / baseline - 1) if baseline > 0 else None
                published = protocol["paper_table_ii_percent"][profile][f"{exo}_{direction}"]
                comparisons.append(
                    {
                        "profile": profile,
                        "exo": exo,
                        "direction": direction,
                        "observed_change_percent": observed,
                        "paper_change_percent": published,
                        "difference_percentage_points": observed - published
                        if observed is not None
                        else None,
                    }
                )
    output.mkdir(parents=True, exist_ok=False)
    for name, records in [("areas", rows), ("table_ii", comparisons)]:
        with (output / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    result = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "parameter_set": protocol["parameter_set"],
        "target_sets": protocol["target_sets"],
        "area_method": "Trapezoidal integral of sampled min_torque(V)<=0 over the selected velocity sign.",
        "scope": "Complete finite-grid comparison; spatial convergence is assessed separately.",
        "areas": rows,
        "table_ii": comparisons,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    _plot(comparisons, output)
    return result


def _plot(comparisons: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    positions = np.arange(len(PROFILES))
    for i, exo in enumerate(["gc", "pd"]):
        for j, direction in enumerate(DIRECTIONS):
            axis = axes[i, j]
            rows = [r for r in comparisons if r["exo"] == exo and r["direction"] == direction]
            actual = [
                r["observed_change_percent"] if r["observed_change_percent"] is not None else np.nan
                for r in rows
            ]
            axis.bar(
                positions - 0.18,
                [r["paper_change_percent"] for r in rows],
                0.36,
                label="Paper Table II",
                color="#9ba8b6",
            )
            axis.bar(positions + 0.18, actual, 0.36, label="Reconciled model", color="#20758d")
            axis.axhline(0, color="#333333", linewidth=0.7)
            axis.set(title=f"{exo.upper()} / {direction} velocity", ylabel="Area change (%)")
            axis.set_xticks(positions, PROFILES)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle("Finite-grid HJB area changes relative to no assistance")
    figure.tight_layout()
    for extension in ["png", "pdf", "svg"]:
        figure.savefig(output / f"table_ii.{extension}", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.protocol, args.out)
    print(json.dumps({"regions": len(result["areas"]), "comparisons": len(result["table_ii"])}, indent=2))


if __name__ == "__main__":
    main()
