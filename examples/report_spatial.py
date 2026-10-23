"""Rebuild the spatial diagnosis from compact projections and frozen provenance.

No MATLAB, GPU, dense 3D histories or local experiment directory is needed.
The two unassisted pilots do not evaluate the GC-effect acceptance criterion.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from examples.diagnose_spatial import section_geometry
from examples.hjb_analytic_reference import exact_bounds
from examples.report_validation import area, overlap, table, write

ROOT = Path(__file__).resolve().parents[1]


def report(root: Path, output: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    frozen = json.loads((ROOT / "experiments/validation_v1.json").read_text())["acceptance"]
    if manifest["acceptance"] != frozen:
        raise ValueError("The original numerical acceptance thresholds must be retained")
    for filename, expected in manifest["diagnostic_sha256"].items():
        if hashlib.sha256((root / "diagnostics" / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Changed diagnostic record or protocol")
    output.mkdir(parents=True, exist_ok=True)
    arrays, summaries, sources, rows = {}, {}, {}, []
    for case in manifest["cases"]:
        path = root / "solves" / case["id"]
        for filename, expected in case["sha256"].items():
            if hashlib.sha256((path / filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Changed evidence: " + case["id"] + "/" + filename)
        for filename, expected in case["preparation_sources"].items():
            path_source = root / "preparation_sources" / (expected + "_" + filename)
            if hashlib.sha256(path_source.read_bytes()).hexdigest() != expected:
                raise ValueError("Changed MATLAB preparation snapshot")
        summary = json.loads((path / "summary.json").read_text())
        if summary["stage"] != "solved":
            raise ValueError("Incomplete solve: " + case["id"])
        source = root / "sources" / (summary["solver_sha256"] + ".py")
        if hashlib.sha256(source.read_bytes()).hexdigest() != summary["solver_sha256"]:
            raise ValueError("Changed solver snapshot")
        sources[case["id"]] = ast.dump(ast.parse(source.read_text()), include_attributes=False)
        data = loadmat(path / "projection.mat")
        field = (data["theta_rad"].ravel(), data["velocity_rad_s"].ravel(), data["projected_value"])
        if not np.isfinite(field[2]).all():
            raise ValueError("Nonfinite projection")
        arrays[case["id"]], summaries[case["id"]] = field, summary
        rows.append(
            {
                "id": case["id"],
                "variant": case["variant"],
                "profile": summary["profile"],
                "requested_resolution": summary.get("requested_resolution", summary["resolution"]),
                "actual_resolution": summary["resolution"],
                "cfl": summary["cfl"],
                "precision": summary["precision"],
                "area": area(*field, summary["direction"]),
                "solver_sha256": summary["solver_sha256"],
            }
        )
    measured = {row["id"]: row for row in rows}
    comparisons = []
    for pair in manifest["comparisons"]:
        coarse, fine = (measured[pair[key]] for key in ["coarse", "fine"])
        a, b = summaries[coarse["id"]], summaries[fine["id"]]
        for key in ["profile", "exo", "direction", "effective_profile", "method", "duration_s", "target"]:
            if a[key] != b[key]:
                raise ValueError("Changed physical comparison contract: " + pair["id"] + "/" + key)
        error = 100 * abs(fine["area"] - coarse["area"]) / fine["area"]
        iou = overlap(arrays[coarse["id"]], arrays[fine["id"]], b["direction"])
        purpose = pair["purpose"]
        if purpose == "temporal_refinement":
            limits = frozen["temporal_area_relative_percent"], frozen["temporal_projection_iou"]
        elif purpose == "precision_guard":
            guard = json.loads((root / "diagnostics/precision_protocol.json").read_text())["acceptance"]
            limits = guard["area_relative_percent"], guard["projection_iou"]
        else:
            limits = frozen["spatial_area_relative_percent"], frozen["common_grid_projection_iou"]
        acceptable = error <= limits[0] and iou >= limits[1]
        spatial = purpose not in {"temporal_refinement", "precision_guard"}
        comparisons.append(
            {
                **pair,
                "area_error_percent": error,
                "projection_iou": iou,
                "area_and_iou_passed": bool(acceptable),
                "area_limit_percent": limits[0],
                "iou_limit": limits[1],
                "same_solver_source": a["solver_sha256"] == b["solver_sha256"],
                "same_solver_ast": sources[coarse["id"]] == sources[fine["id"]],
                "assistance_effect_change_pp": None,
                "spatial_gate": ("INCOMPLETE_GC" if acceptable else "NOT_MET") if spatial else None,
                "other_gate": ("PASSED" if acceptable else "FAILED") if not spatial else None,
            }
        )
    geometry = []
    for item in manifest["geometry"]:
        if (
            hashlib.sha256((root / "geometry_inputs" / item["file"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError("Changed goal-geometry evidence")
        with np.load(root / "geometry_inputs" / item["file"]) as sections:
            geometry.extend(
                dict(case=item["case"], representation=item["representation"], **row)
                for row in section_geometry(sections)
            )
    table(output / "areas.csv", rows, lineterminator="\n")
    table(output / "comparisons.csv", comparisons, lineterminator="\n")
    table(output / "initial_geometry.csv", geometry, lineterminator="\n")

    for filename, expected in manifest["analytic_sha256"].items():
        if hashlib.sha256((root / "analytic_reference" / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Changed analytic-reference evidence")
    analytic = json.loads((root / "analytic_reference/summary.json").read_text())
    expected = {(n, flux) for n in analytic["resolutions"] for flux in ["lax_friedrichs", "godunov"]}
    actual = [(case["resolution"], case["flux"]) for case in analytic["cases"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Incomplete analytic-reference comparison")
    analytical_rows = []
    for case in analytic["cases"]:
        with np.load(root / "analytic_reference" / case["field"]) as source:
            coordinates = [np.linspace(-0.25, 0.25, 601), np.linspace(-0.75, 0.75, 601)]
            x, v = np.meshgrid(*coordinates, indexing="ij")
            lower, upper, feasible = exact_bounds(v)
            exact = feasible & (x >= lower) & (x <= upper)
            value = RegularGridInterpolator((source["position"], source["velocity"]), source["value"])(
                np.stack([x, v], -1)
            )
            inside = value <= 0
            actual_area = float(
                np.trapezoid(np.trapezoid(inside.astype(float), coordinates[1], axis=1), coordinates[0])
            )
            analytical_rows.append(
                {
                    **case,
                    "area": actual_area,
                    "area_error_percent": 100
                    * abs(actual_area - analytic["exact_common_grid_area"])
                    / analytic["exact_common_grid_area"],
                    "iou": float(np.count_nonzero(inside & exact) / np.count_nonzero(inside | exact)),
                }
            )
    table(output / "analytic_reference.csv", analytical_rows, lineterminator="\n")
    result = {
        "solves": len(rows),
        "fine_grid_runs_complete": manifest["fine_campaign_complete"],
        "comparisons": comparisons,
        "analytic_reference": analytical_rows,
        "acceptance": frozen,
        "spatial_gate": "NOT_ACCEPTED",
        "scope": "Unassisted diagnostic pilots; no new GC-effect or all-family acceptance is asserted.",
    }
    write(output / "summary.json", result)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, profile in zip(axes[:2], ["YF", "OM"], strict=True):
        for variant, label in [
            ("original", "Original LF"),
            ("aligned_godunov", "Aligned goal + Godunov"),
            ("aligned_weno5", "Aligned goal + WENO5"),
        ]:
            selected = sorted(
                [r for r in rows if r["profile"] == profile and r["variant"] == variant],
                key=lambda row: row["requested_resolution"][0],
            )
            axis.plot(
                [r["requested_resolution"][0] for r in selected],
                [r["area"] for r in selected],
                marker="o",
                label=label,
            )
        axis.set(
            title=profile + " · forward, no assistance",
            xlabel="Requested nodes per axis",
            ylabel="Projected area [rad²/s]",
        )
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    for flux, label in [("lax_friedrichs", "LF"), ("godunov", "Godunov")]:
        points = [row for row in analytical_rows if row["flux"] == flux]
        axes[2].plot(
            [row["resolution"] for row in points],
            [row["area_error_percent"] for row in points],
            marker="o",
            label=label,
        )
    axes[2].set(
        title="Independent double integrator",
        xlabel="Nodes per active axis",
        ylabel="Area error vs analytic set [%]",
    )
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.savefig(output / "spatial_diagnosis.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps({"solves": len(rows), "comparisons": len(comparisons), "status": result["spatial_gate"]})
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "docs/results/spatial_v2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.out)
