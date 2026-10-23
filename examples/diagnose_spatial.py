"""Rebuild initial-goal geometry and quadrature diagnostics from actual fields."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from scipy.optimize import brentq

from examples.report_validation import area, table, write


def section_geometry(sections):
    records = []
    for axis, width in enumerate(sections["widths"]):
        coordinates = sections[f"axis{axis}"]
        field = sections[f"field{axis}"]

        def value(displacement):
            return float(np.interp(displacement, coordinates, field))

        lower, upper = brentq(value, -2 * width, 0), brentq(value, 0, 2 * width)
        records.append(
            dict(
                axis=["theta", "velocity", "torque"][axis],
                physical_half_width=float(width),
                represented_lower=lower,
                represented_upper=upper,
                width_error_percent=100 * (1 - (upper - lower) / (2 * width)),
            )
        )
    return records


def geometry(problem, section_file=None):
    with h5py.File(problem, "r") as source:
        metadata = json.loads("".join(chr(int(c)) for c in np.asarray(source["problem_json"]).ravel()))
        box = metadata["summary"]["target"]
        axes = [np.asarray(source[k]).ravel() for k in ["theta_rad", "velocity_rad_s", "human_torque_nm"]]
        interpolator = RegularGridInterpolator(axes, np.asarray(source["initial_value"]).T)
        center = np.array(box["ankleEq"])
        widths = [box[k] for k in ["thetaAnkRange", "thetaDotAnkRange", "torqueAnkRange"]]
        sections = {"widths": np.array(widths), "center": center}
        for axis in range(3):
            points = np.broadcast_to(center, (len(axes[axis]), 3)).copy()
            points[:, axis] = axes[axis]
            sections[f"axis{axis}"] = axes[axis] - center[axis]
            sections[f"field{axis}"] = interpolator(points)
    if section_file is not None:
        section_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(section_file, **sections)
    return section_geometry(sections)


def diagnose(baseline, root, output):
    output.mkdir(parents=True, exist_ok=True)
    geometry_rows, quadrature_rows = [], []
    for profile, target in [("YF", "source_entry"), ("OM", "common_ankle")]:
        for n in [151, 225]:
            stem = f"{profile}_none_forward_{target}_mt100_mrtd100_n{n}"
            for label, parent in [
                ("original", baseline / "prepared"),
                ("normalized", root / "normalized_prepared"),
                ("normalized_aligned", root / "aligned_prepared"),
            ]:
                geometry_rows.extend(
                    dict(case=stem, representation=label, **row)
                    for row in geometry(
                        parent / stem / "problem.mat", output / "geometry_inputs" / f"{stem}_{label}.npz"
                    )
                )
            data = loadmat(baseline / "solved" / (stem + "_cfl0.8") / "projection.mat")
            axes = (data["theta_rad"].ravel(), data["velocity_rad_s"].ravel(), data["projected_value"])
            x, y = np.meshgrid(np.linspace(0.65, 1.8, 1201), np.linspace(0, 2.85, 1201), indexing="ij")
            mask = RegularGridInterpolator(axes[:2], axes[2])(np.stack([x, y], -1)) <= 0
            common = float(np.trapezoid(np.trapezoid(mask.astype(float), y[0], axis=1), x[:, 0]))
            quadrature_rows.append(dict(case=stem, native_area=area(*axes, "forward"), common_area=common))
    table(output / "initial_geometry.csv", geometry_rows, lineterminator="\n")
    table(output / "quadrature.csv", quadrature_rows, lineterminator="\n")
    write(output / "diagnostics.json", dict(geometry=geometry_rows, quadrature=quadrature_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("logs/validation_v1/hjb"))
    parser.add_argument("--root", type=Path, default=Path("logs/spatial_convergence"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.baseline, args.root, args.out)
