"""Export every moving-grid feedback slice and verify the accepted final SI projection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat, savemat

from exobalancelab.hjb_transport import integrate, load_problem, paper_problem, source_hashes
from exobalancelab.paper import Profile
from exobalancelab.reachability import policy_contract
from exobalancelab.transport_policy import FORMAT, switching_codes


@jax.jit
def native_switching_codes(value):
    """Transfer only the exact int8 decisions from the solver device."""
    left = jnp.concatenate([value[..., :1], value[..., :-2], value[..., -2:-1]], axis=-1)
    right = jnp.concatenate([value[..., 1:2], value[..., 2:], value[..., -1:]], axis=-1)
    return (right > left).astype(jnp.int8) - (right < left).astype(jnp.int8)


def device_switching_codes(value):
    """Bound device workspace while collecting exact native decisions on the host."""
    codes = np.empty(value.shape, dtype=np.int8)
    for first in range(0, value.shape[0], 32):
        last = min(first + 32, value.shape[0])
        codes[first:last] = np.asarray(native_switching_codes(value[first:last]))
    return codes


def export(source, destination, reference, resolution=673, step=0.01):
    if destination.exists():
        raise FileExistsError(destination)
    inputs = load_problem(source)
    summary = inputs["summary"]
    problem = paper_problem(summary, inputs["lower"], inputs["upper"])
    boxes = []
    for name, center in [("Ank", "ankleEq"), ("Toe", "toeEq")]:
        center = np.array(summary["target"][center])
        widths = np.array(
            [summary["target"][f"{axis}{name}Range"] for axis in ["theta", "thetaDot", "torque"]]
        )
        if not any(np.array_equal(center, c) and np.array_equal(widths, w) for c, w, _ in boxes):
            boxes.append((center, widths, min(widths * inputs["state_scale"])))
    destination.mkdir(parents=True)
    (destination / "problem.json").write_text(json.dumps(inputs, sort_keys=True, indent=2) + "\n")
    count = int(np.ceil(summary["duration_s"] / step))
    times = np.linspace(0, summary["duration_s"], count + 1)
    profile = Profile(name=summary["profile"], **summary["effective_profile"])
    started = time.perf_counter()
    projection = None
    metadata = dict(
        summary,
        stage="running",
        resolution=resolution,
        integration_step_s=float(times[1]),
        sources=source_hashes(),
        input_sha256=hashlib.sha256((destination / "problem.json").read_bytes()).hexdigest(),
        policy_format=FORMAT,
        decision_reference_checks="Full native initial/final fields for each goal",
        control_interpolation="trilinear in state; linear in remaining time",
        goal_selection="minimum final value at episode start, held for that episode",
    )
    for path in [Path(__file__), Path("exobalancelab/transport_policy.py")]:
        metadata["sources"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "sources").mkdir()
    for name in metadata["sources"]:
        folder = "examples" if name == "export_transport_policy.py" else "exobalancelab"
        (destination / "sources" / name).write_bytes((Path(folder) / name).read_bytes())

    def record():
        (destination / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    record()
    with h5py.File(destination / "policy.h5", "w") as file:
        file.attrs.update(
            format=FORMAT, complete=False, contract_json=json.dumps(policy_contract(profile, summary["exo"]))
        )
        file.create_dataset("time_to_go_s", data=times)
        file.create_dataset("rate_bounds_nm_s", data=[profile.rate_min_nm_s, profile.rate_max_nm_s])
        for goal_index, box in enumerate(boxes):
            group = file.create_group(f"goals/{goal_index}")
            axis_arrays = {
                name: group.create_dataset(name, (count + 1, resolution), dtype="f8")
                for name in ["theta_rad", "velocity_rad_s", "human_torque_nm"]
            }

            def snapshot(t, axes, value):
                index = round(t / times[1])
                codes = device_switching_codes(value)
                if index in (0, count):
                    array = np.asarray(value)
                    if not np.array_equal(codes, switching_codes(array)):
                        raise ValueError("Device decisions differ from the native CPU derivative signs")
                group.create_dataset(
                    f"controls/{index:04d}",
                    data=codes,
                    chunks=(16, 16, 64),
                    compression="gzip",
                    compression_opts=1,
                )
                for dataset, axis in zip(axis_arrays.values(), axes, strict=True):
                    dataset[index] = np.asarray(axis)
                if index == count:
                    group.create_dataset(
                        "final_value",
                        data=array,
                        chunks=(16, 16, 64),
                        compression="gzip",
                        compression_opts=1,
                        shuffle=True,
                    )
                if index % 10 == 0:
                    file.flush()
                    print(
                        json.dumps(dict(goal=goal_index, time_s=t, wall_s=time.perf_counter() - started)),
                        flush=True,
                    )

            axes, final = integrate(
                problem, [box], (resolution,) * 3, summary["duration_s"], step, snapshot=snapshot
            )
            current = final.min(axis=2)
            projection = current if projection is None else np.minimum(projection, current)
            del final
        actual = loadmat(reference / "projection.mat")
        equal = np.array_equal(projection, actual["projected_value"]) and all(
            np.array_equal(a, actual[name].ravel())
            for a, name in zip(axes[:2], ["theta_rad", "velocity_rad_s"], strict=True)
        )
        # The accepted projection stores angle and velocity only.
        if not equal:
            raise ValueError("History export changed the accepted final projection")
        file.attrs["complete"] = True
    savemat(
        destination / "projection.mat",
        dict(theta_rad=axes[0], velocity_rad_s=axes[1], projected_value=projection),
        do_compression=True,
    )
    with (destination / "policy.h5").open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    metadata.update(
        stage="exported",
        wall_s=time.perf_counter() - started,
        policy_sha256=digest,
        policy_bytes=(destination / "policy.h5").stat().st_size,
        accepted_projection_identical=True,
        reference_projection_sha256=hashlib.sha256((reference / "projection.mat").read_bytes()).hexdigest(),
    )
    record()
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.problem, args.out, args.reference), indent=2))
