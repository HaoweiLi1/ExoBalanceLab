"""Rechunk the completed native history without changing any values or metadata."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

source = Path("logs/hjb_OF_theta201_buffer05_dt001")
output = Path("logs/hjb_native201_io")
target = output / "policy.partial.mat"
assert not target.exists() and not (output / "policy.mat").exists()
started = time.perf_counter()
digest = hashlib.sha256()
with h5py.File(source / "policy.mat") as original:
    userblock = original.userblock_size
    with h5py.File(target, "w", userblock_size=userblock) as packed:
        packed.attrs.update(original.attrs)
        for key in original:
            if key != "value":
                original.copy(key, packed)
        values = original["value"]
        chunks = (1, values.shape[1], 1, values.shape[3])
        result = packed.create_dataset(
            "value", values.shape, dtype=values.dtype, chunks=chunks,
            compression="gzip", compression_opts=1, shuffle=True,
        )
        result.attrs.update(values.attrs)
        for velocity in range(values.shape[2]):
            slab = values[:, :, velocity, :]
            result[:, :, velocity, :] = slab
            assert np.array_equal(slab, result[:, :, velocity, :])
            digest.update(slab.tobytes(order="C"))
        record = {
            "source": str(source / "policy.mat"),
            "source_chunks": list(values.chunks),
            "result_chunks": list(chunks),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "every_value_equal": True,
            "value_sha256_velocity_time_torque_theta_order": digest.hexdigest(),
            "scope": "Storage transformation only; original MATLAB solve and file retained.",
        }
if userblock:
    with (source / "policy.mat").open("rb") as original, target.open("r+b") as packed:
        packed.write(original.read(userblock))
target.rename(output / "policy.mat")
for name in ["summary.json", "projection.mat"]:
    shutil.copy2(source / name, output / name)
for label, folder in [("source", source), ("result", output)]:
    with (folder / "policy.mat").open("rb") as file:
        record[f"{label}_file_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
record["wall_s"] = time.perf_counter() - started
(output / "storage_transform.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
