"""Recheck the deliberately insufficient coarse reference without JAX or a GPU."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from examples.benchmark_transport import triple_bounds
from examples.transport_convergence import protocol

folder = Path(__file__).resolve().parent
manifest = json.loads((folder / "manifest.json").read_text())
for name, digest in manifest["sources"].items():
    assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
assert hashlib.sha256(Path(inspect.getsourcefile(triple_bounds)).read_bytes()).hexdigest() == manifest["sources"]["benchmark_transport.py"]
source = folder / manifest["field"]
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
x, v = np.linspace(-.1, .1, 601), np.linspace(-.3, .3, 601)
xx, vv = np.meshgrid(x, v, indexing="ij")
lower, upper, feasible = triple_bounds(vv)
exact = feasible & (xx >= lower) & (xx <= upper)
with np.load(source) as field:
    interpolated = RegularGridInterpolator((field["position"], field["velocity"]), field["value"], bounds_error=False, fill_value=1)(np.stack([xx, vv], -1))
numeric = interpolated <= 0

def area(mask):
    return float(np.trapezoid(np.trapezoid(mask.astype(float), v, axis=1), x))

error = 100 * abs(area(numeric) / area(exact) - 1)
iou = float((numeric & exact).sum() / (numeric | exact).sum())
limits = protocol()["analytic_acceptance"]
result = dict(resolution=manifest["resolution"], time_step_s=manifest["time_step_s"], area_error_percent=error, projection_iou=iou, passed=error <= limits["area_relative_percent"] and iou >= limits["projection_iou"])
(folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
