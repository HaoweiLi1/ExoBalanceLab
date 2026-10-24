"""Compare complete HJB projections with Table II or across spatial grids.

Inputs are recorded solver summaries and SI theta/velocity value projections.
Outputs retain sampled areas in rad²/s, common-domain comparisons and provenance.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from exobalancelab.paper import AUTHORS_PROFILES

PROFILES = ("YM", "YF", "OM", "OF", "wOF")
MODES = ("none", "gc", "pd")
DIRECTIONS = ("forward", "backward")


def _require_solved(cases: list[dict]) -> None:
    missing = []
    for case in cases:
        path = Path(case["solved"])
        if not (path / "summary.json").exists():
            missing.append(str(path))
        elif json.loads((path / "summary.json").read_text()).get("stage") != "solved":
            missing.append(str(path))
    if missing:
        raise ValueError("Incomplete HJB campaign: " + ", ".join(missing))


def _area(theta: np.ndarray, velocity: np.ndarray, inside: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(inside.astype(float), velocity, axis=1), theta))


def _case_data(case: dict, protocol: dict) -> tuple[dict, dict, tuple[np.ndarray, ...]]:
    path = Path(case["solved"])
    summary = json.loads((path / "summary.json").read_text())
    profile = AUTHORS_PROFILES[case["profile"]]
    for key in ["mt_scale", "mrtd_scale"]:
        if summary.get(key, 1) != case.get(key, 1):
            raise ValueError(f"Different {key} in {path}")
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
    resolution = (
        protocol["requested_resolution"]
        if "requested_resolution" in protocol
        else case["requested_resolution"]
    )
    for key, expected_value in [
        ("duration_s", protocol["horizon_s"]),
        ("output_step_s", protocol["output_step_s"]),
        ("velocity_buffer_rad_s", protocol["velocity_buffer_rad_s"]),
        ("requested_resolution", resolution),
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
        len(theta) < 2
        or len(velocity) < 2
        or field.shape != (len(theta), len(velocity))
        or list(field.shape) != summary["resolution"][:2]
        or not np.isfinite(field).all()
        or not np.isfinite(theta).all()
        or not np.isfinite(velocity).all()
        or np.any(np.diff(theta) <= 0)
        or np.any(np.diff(velocity) <= 0)
    ):
        raise ValueError(f"Invalid projected value grid in {path}")
    selected = velocity >= 0 if case["direction"] == "forward" else velocity <= 0
    row = {
        "profile": case["profile"],
        "exo": case["exo"],
        "direction": case["direction"],
        "area_rad2_per_s": _area(theta, velocity[selected], field[:, selected] <= 0),
        "theta_min_rad": float(theta[0]),
        "theta_max_rad": float(theta[-1]),
        "velocity_min_rad_s": float(velocity[0]),
        "velocity_max_rad_s": float(velocity[-1]),
        "requested_resolution": "x".join(map(str, resolution)),
        "resolution": "x".join(map(str, summary["resolution"])),
        "wall_s": summary["wall_s"],
        "run": str(path),
        "projection_sha256": hashlib.sha256(projection.read_bytes()).hexdigest(),
    }
    return row, summary, (theta, velocity, field)


def _csv(output: Path, name: str, records: list[dict]) -> None:
    with (output / f"{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


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
    _require_solved(cases)
    rows = [_case_data(case, protocol)[0] for case in cases]
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
        _csv(output, name, records)
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


def compare_grids(protocol_path: Path, output: Path) -> dict:
    """Compare declared mode/direction pairs for one profile across recorded grids.

    Coarse values are linearly interpolated onto finer nodes in the common domain.
    Native-grid and common-grid area differences remain separate measurements.
    All six pairs are required unless the protocol explicitly declares a subset.
    """
    protocol = json.loads(protocol_path.read_text())
    if protocol["parameter_set"] != "authors":
        raise ValueError("This study requires the pinned author parameter set")
    cases = protocol["cases"]
    profiles = {c["profile"] for c in cases}
    resolutions = sorted({tuple(size) for size in protocol["requested_resolutions"]})
    if len(profiles) != 1 or len(resolutions) < 2:
        raise ValueError("Grid study requires one profile and at least two resolutions")
    for lower, upper in zip(resolutions, resolutions[1:]):
        if len(lower) != 3 or len(upper) != 3 or np.any(np.array(upper) < lower):
            raise ValueError("Requested grids must refine each axis without coarsening another")
    all_pairs = [(mode, direction) for mode in MODES for direction in DIRECTIONS]
    pairs = [tuple(pair) for pair in protocol.get("mode_direction_pairs", all_pairs)]
    if not pairs or len(set(pairs)) != len(pairs) or not set(pairs).issubset(all_pairs):
        raise ValueError("Grid study requires distinct, valid mode/direction pairs")
    keys = [(c["exo"], c["direction"], tuple(c["requested_resolution"])) for c in cases]
    expected = {(mode, direction, size) for mode, direction in pairs for size in resolutions}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Grid study requires every declared mode/direction pair at every resolution")
    _require_solved(cases)
    loaded = {key: _case_data(case, protocol) for key, case in zip(keys, cases, strict=True)}
    comparisons = []
    for mode, direction in pairs:
        for lower, upper in zip(resolutions, resolutions[1:]):
            a, sa, (xa, va, fa) = loaded[mode, direction, lower]
            b, sb, (xb, vb, fb) = loaded[mode, direction, upper]
            if set(sa["target"]) != set(sb["target"]) or any(
                not np.allclose(sa["target"][key], sb["target"][key], rtol=1e-12, atol=1e-12)
                for key in sa["target"]
            ):
                raise ValueError(f"Different target sets for {mode}/{direction}")
            if any(sa[key] != sb[key] for key in ["solver_sha256", "precision"]):
                raise ValueError(f"Different solver or precision for {mode}/{direction}")
            theta_mask = (xb >= xa[0]) & (xb <= xa[-1])
            velocity_mask = (vb >= va[0]) & (vb <= va[-1])
            velocity_mask &= vb >= 0 if direction == "forward" else vb <= 0
            theta, velocity = xb[theta_mask], vb[velocity_mask]
            if len(theta) < 2 or len(velocity) < 2:
                raise ValueError(f"Insufficient common domain for {mode}/{direction}")
            points = np.stack(np.meshgrid(theta, velocity, indexing="ij"), axis=-1)
            coarse = RegularGridInterpolator((xa, va), fa)(points) <= 0
            fine = fb[np.ix_(theta_mask, velocity_mask)] <= 0
            union = np.count_nonzero(coarse | fine)
            comparisons.append(
                {
                    "profile": a["profile"],
                    "exo": mode,
                    "direction": direction,
                    "coarse_requested_resolution": a["requested_resolution"],
                    "fine_requested_resolution": b["requested_resolution"],
                    "native_grid_area_change_percent": 100 * (b["area_rad2_per_s"] / a["area_rad2_per_s"] - 1)
                    if a["area_rad2_per_s"]
                    else None,
                    "common_coarse_area_rad2_per_s": _area(theta, velocity, coarse),
                    "common_fine_area_rad2_per_s": _area(theta, velocity, fine),
                    "common_sign_disagreement_nodes": int(np.count_nonzero(coarse != fine)),
                    "common_negative_set_jaccard": float(np.count_nonzero(coarse & fine) / union)
                    if union
                    else 1.0,
                    "common_grid_nodes": int(coarse.size),
                    "common_theta_min_rad": float(theta[0]),
                    "common_theta_max_rad": float(theta[-1]),
                    "common_velocity_min_rad_s": float(velocity[0]),
                    "common_velocity_max_rad_s": float(velocity[-1]),
                }
            )
    rows = [loaded[key][0] for key in keys]
    result = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "parameter_set": protocol["parameter_set"],
        "target_sets": protocol["target_sets"],
        "mode_direction_pairs": pairs,
        "scope": "Spatial sensitivity on recorded grids; no automatic convergence certificate.",
        "comparison_method": (
            "Linear interpolation of coarse projected values onto finer common-domain nodes, "
            "then threshold at zero; no extrapolation."
        ),
        "area_method": (
            "Trapezoidal integration of the sampled nonpositive indicator. Native-grid areas "
            "also reflect quadrature and domain differences."
        ),
        "areas": rows,
        "comparisons": comparisons,
    }
    output.mkdir(parents=True, exist_ok=False)
    _csv(output, "areas", rows)
    _csv(output, "comparisons", comparisons)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    _plot_grid_study(rows, resolutions, output)
    return result


def _plot_grid_study(rows: list[dict], resolutions: list[tuple], output: Path) -> None:
    import matplotlib.pyplot as plt

    directions = [direction for direction in DIRECTIONS if any(r["direction"] == direction for r in rows)]
    figure, axes = plt.subplots(
        1, len(directions), figsize=(5 * len(directions), 4.5), sharex=True, squeeze=False
    )
    labels = ["x".join(map(str, size)) for size in resolutions]
    for axis, direction in zip(axes[0], directions, strict=True):
        for mode, color in zip(MODES, ["#24486d", "#238f83", "#d58d32"], strict=True):
            lookup = {
                r["requested_resolution"]: r["area_rad2_per_s"]
                for r in rows
                if r["exo"] == mode and r["direction"] == direction
            }
            if not lookup:
                continue
            axis.plot(
                range(len(labels)),
                [lookup[label] for label in labels],
                "o-",
                label="No assistance" if mode == "none" else mode.upper(),
                color=color,
            )
        axis.set_xticks(range(len(labels)), labels, fontsize=8)
        axis.set(
            title=f"{direction.capitalize()} velocity",
            xlabel="Requested grid",
            ylabel="Projected area (rad²/s)",
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f"{rows[0]['profile']}: projected-area sensitivity to grid spacing")
    figure.tight_layout()
    for extension in ["png", "pdf", "svg"]:
        figure.savefig(output / f"grid_study.{extension}", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--grid-study", action="store_true", help="Compare a complete spatial-resolution study"
    )
    args = parser.parse_args()
    result = compare_grids(args.protocol, args.out) if args.grid_study else report(args.protocol, args.out)
    count = len(result["comparisons"] if args.grid_study else result["table_ii"])
    print(json.dumps({"regions": len(result["areas"]), "comparisons": count}, indent=2))


if __name__ == "__main__":
    main()
