"""Represent the same physical terminal boxes without unequal-width clipping.

The original max(|x-c|-w) has depth min(w). On a coarse grid that truncates
the wider-axis zero crossings. The width-normalized infinity norm has the
same analytic zero set and nodal signs. Multiplying by min(w) retains the
original center depth and obstacle units. This changes the implicit function,
not the physical goal, dynamics, contact constraints, or horizon.
"""

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


def box_value(axes, center, widths):
    """Width-normalized box field in the supplied coordinate system."""
    shape = tuple(len(a) for a in axes)
    value = np.full(shape, -np.inf)
    for dimension, (axis, c, w) in enumerate(zip(axes, center, widths, strict=True)):
        if not np.isfinite(w) or w <= 0:
            raise ValueError("Box half-widths must be finite and positive")
        view = [1] * len(axes)
        view[dimension] = len(axis)
        value = np.maximum(value, (np.abs(axis - c) / w - 1).reshape(view))
    return value * min(widths)


def prepare(source: Path, output: Path) -> dict:
    """Re-encode frozen MATLAB boxes on exactly the original grid."""
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(source, "r") as original:
        metadata = json.loads("".join(chr(int(c)) for c in np.asarray(original["problem_json"]).ravel()))
        scale = np.asarray(original["state_scale"]).ravel()
        axes = [
            np.asarray(original[k]).ravel() * s
            for k, s in zip(["theta_rad", "velocity_rad_s", "human_torque_nm"], scale, strict=True)
        ]
        target = metadata["summary"]["target"]
        fields = []
        for label, key in [("Ank", "ankleEq"), ("Toe", "toeEq")]:
            widths = (
                np.array(
                    [
                        target[f"theta{label}Range"],
                        target[f"thetaDot{label}Range"],
                        target[f"torque{label}Range"],
                    ]
                )
                * scale
            )
            fields.append(box_value(axes, np.array(target[key]) * scale, widths))
        value = np.maximum(np.minimum(*fields), np.asarray(original["obstacle_value"]).T)
        before = np.asarray(original["initial_value"]).T
        if not np.array_equal(value < 0, before < 0) or not np.array_equal(value <= 0, before <= 0):
            raise ValueError("Target re-encoding changed a nodal sign")
        with source.open("rb") as file:
            source_hash = hashlib.file_digest(file, "sha256").hexdigest()
        metadata["summary"].update(
            target_field="width_normalized",
            parent_problem_sha256=source_hash,
            geometry_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            target_nodal_signs_identical=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    with h5py.File(output, "r+") as result:
        result["initial_value"][...] = value.T
        del result["problem_json"]
        result["problem_json"] = np.array([ord(c) for c in json.dumps(metadata)], dtype=np.uint16)[:, None]
    return metadata["summary"]
