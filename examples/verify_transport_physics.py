"""Check the SI transport dynamics against frozen, MATLAB-prepared fields."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from examples.report_validation import write
from examples.transport_convergence import ROOT, family_name, protocol


def verify(prepared, output):
    from exobalancelab.hjb_transport import load_problem, paper_problem, physical_fields, source_hashes

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = []
    for case in protocol()["families"]:
        name = family_name(case)
        source = prepared / (name + "_n225x225x225") / "problem.mat"
        inputs = load_problem(source)
        original = json.loads(
            (ROOT / "docs/results/validation_v1/hjb/solved" / case["id"] / "summary.json").read_text()
        )
        for key in ["target", "effective_profile", "method", "duration_s", "target_definition"]:
            if inputs["summary"][key] != original[key]:
                raise ValueError("Changed original physical problem: " + name + "/" + key)
        with h5py.File(source) as data:
            axes = [np.asarray(data[k]).ravel() for k in ["theta_rad", "velocity_rad_s", "human_torque_nm"]]
            scale = np.asarray(data["state_scale"]).ravel()
            time_scale = float(np.asarray(data["time_scale_s"]).item())
            index = np.column_stack(
                [np.random.default_rng(20261023 + i).integers(0, len(a), 1000) for i, a in enumerate(axes)]
            )
            states = [a[index[:, i]] for i, a in enumerate(axes)]
            expected = [
                np.array([data[key][k, j, i] for i, j, k in index])
                for key in ["drift_theta", "drift_velocity", "obstacle_value"]
            ]
        expected[0] /= time_scale
        expected[1] /= time_scale * scale[1]
        p = paper_problem(inputs["summary"], np.array(inputs["lower"]), np.array(inputs["upper"]))
        drift, obstacle = physical_fields(p, states)
        actual = [np.asarray(drift[0]), np.asarray(drift[1]), np.asarray(obstacle)]
        errors = [float(np.max(abs(a - b))) for a, b in zip(actual, expected)]
        row = dict(
            family=name,
            samples=len(index),
            max_abs_errors=errors,
            passed=max(errors) < 1e-12,
            prepared_sha256=inputs["prepared_sha256"],
            original_physical_contract_unchanged=True,
        )
        target = inputs["summary"]["target"]
        boxes = [
            (
                np.array(target[key]),
                np.array([target[f"{a}{label}Range"] for a in ["theta", "thetaDot", "torque"]]),
            )
            for label, key in [("Ank", "ankleEq"), ("Toe", "toeEq")]
        ]
        shared_lower = np.maximum(inputs["lower"], np.min([c - 2 * w for c, w in boxes], axis=0))
        shared_upper = np.minimum(inputs["upper"], np.max([c + 2 * w for c, w in boxes], axis=0))
        goal_sampling = []
        for center, widths in boxes:
            counts = {}
            for label, lower, upper in [
                ("shared_2x", shared_lower, shared_upper),
                (
                    "separate_2x",
                    np.maximum(inputs["lower"], center - 2 * widths),
                    np.minimum(inputs["upper"], center + 2 * widths),
                ),
            ]:
                axes = [np.linspace(lo, hi, 225) for lo, hi in zip(lower, upper)]
                counts[label] = int(
                    np.prod([np.count_nonzero(abs(axis - c) < w) for axis, c, w in zip(axes, center, widths)])
                )
            goal_sampling.append(counts)
        row["initial_negative_goal_nodes_before_contact"] = goal_sampling
        rows.append(row)
        sample_file = output / (name + ".npz")
        np.savez_compressed(sample_file, states=np.array(states), expected=np.array(expected))
        row["sample_sha256"] = hashlib.sha256(sample_file.read_bytes()).hexdigest()
        print(json.dumps(row), flush=True)
        write(
            output / "summary.json",
            dict(
                sources=source_hashes(),
                tolerance_abs=1e-12,
                cases=rows,
                passed=all(r["passed"] for r in rows),
            ),
        )
    return all(r["passed"] for r in rows)


def replay(inputs_directory, samples_directory, output):
    """Recheck frozen MATLAB samples without MATLAB or the dense preparation files."""
    from exobalancelab.hjb_transport import load_problem, paper_problem, physical_fields, source_hashes

    if output.exists():
        raise FileExistsError(output)
    recorded = json.loads((samples_directory / "summary.json").read_text())
    expected_names = [family_name(case) for case in protocol()["families"]]
    if [case["family"] for case in recorded["cases"]] != expected_names:
        raise ValueError("The full original physical cohort is required")
    rows = []
    for case, row in zip(protocol()["families"], recorded["cases"]):
        name = row["family"]
        inputs = load_problem(inputs_directory / (name + ".json"))
        original = json.loads(
            (ROOT / "docs/results/validation_v1/hjb/solved" / case["id"] / "summary.json").read_text()
        )
        for key in ["target", "effective_profile", "method", "duration_s", "target_definition"]:
            if inputs["summary"][key] != original[key]:
                raise ValueError("Changed original physical problem: " + name + "/" + key)
        sample_file = samples_directory / (name + ".npz")
        if (
            hashlib.sha256(sample_file.read_bytes()).hexdigest() != row["sample_sha256"]
            or inputs["prepared_sha256"] != row["prepared_sha256"]
        ):
            raise ValueError("Changed frozen MATLAB evidence: " + name)
        with np.load(sample_file) as sample:
            states, expected = sample["states"], sample["expected"]
        problem = paper_problem(inputs["summary"], np.array(inputs["lower"]), np.array(inputs["upper"]))
        drift, obstacle = physical_fields(problem, states)
        actual = np.array([np.asarray(drift[0]), np.asarray(drift[1]), np.asarray(obstacle)])
        errors = np.max(abs(actual - expected), axis=1).tolist()
        rows.append(dict(row, max_abs_errors=errors, passed=max(errors) < 1e-12))
    result = dict(
        sources=source_hashes(), tolerance_abs=1e-12, cases=rows, passed=all(r["passed"] for r in rows)
    )
    write(output / "summary.json", result)
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--prepared", type=Path)
    selection.add_argument("--samples", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples:
        if args.inputs is None:
            parser.error("--samples requires --inputs")
        passed = replay(args.inputs, args.samples, args.out)
    else:
        passed = verify(args.prepared, args.out)
    raise SystemExit(0 if passed else 1)
