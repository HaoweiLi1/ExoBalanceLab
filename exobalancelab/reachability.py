"""Replay exported MATLAB/helperOC HJB feedback on the paper's lifted state.

Coordinates: absolute theta [rad], angular velocity [rad/s], HUMAN torque [Nm],
remaining horizon [s]. Value gradients select human torque RATE [Nm/s], Eq. (11).
A sampled recovery controller is never substituted for missing HJB evidence.
"""

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from exobalancelab.paper import Profile

# SI conversion can place an exported time endpoint a few floating-point ulps below it.
TIME_TOLERANCE_S = 1e-12


class PolicyCoverageError(ValueError):
    """A query outside the exported domain, retaining SI coordinates and bounds."""

    def __init__(self, point: list[float], axes: tuple[np.ndarray, ...]) -> None:
        names = ["theta_rad", "velocity_rad_s", "human_torque_nm", "remaining_s"]
        self.query = dict(zip(names, map(float, point), strict=True))
        self.bounds = {
            name: [float(axis[0]), float(axis[-1])] for name, axis in zip(names, axes, strict=True)
        }
        violations = [
            f"{name}={value:.12g} outside [{self.bounds[name][0]:.12g}, {self.bounds[name][1]:.12g}]"
            for name, value in self.query.items()
            if not self.bounds[name][0] <= value <= self.bounds[name][1]
        ]
        super().__init__("HJB query out of bounds: " + "; ".join(violations))


def policy_contract(profile: Profile, exo: str) -> dict:
    """Describe the paper plant for a policy, including effective MT/MRTD limits."""
    parameters = asdict(profile)
    parameters.pop("name")
    return {"model": "paper", "exo": exo, "profile": parameters}


class _HDF5Gradient:
    """Interpolate two adjacent time slices without loading a full 4D policy."""

    def __init__(
        self, path: Path, axes: tuple[np.ndarray, ...], derivatives: np.ndarray | None = None
    ) -> None:
        self.path = path
        self.grid = axes
        self._derivatives = derivatives
        self._slices: OrderedDict[int, RegularGridInterpolator] = OrderedDict()

    def _slice(self, index: int) -> RegularGridInterpolator:
        import h5py

        if index not in self._slices:
            if self._derivatives is None:
                with h5py.File(self.path, "r") as source:
                    value = np.asarray(source["value"][index], dtype=float).T
                derivative = np.gradient(value, self.grid[2], axis=2, edge_order=1)
            else:
                derivative = self._derivatives[index].T
            self._slices[index] = RegularGridInterpolator(self.grid[:3], derivative, bounds_error=True)
            if len(self._slices) > 2:
                self._slices.popitem(last=False)
        self._slices.move_to_end(index)
        return self._slices[index]

    def __call__(self, point: list[float]) -> np.ndarray:
        times = self.grid[3]
        upper = int(np.clip(np.searchsorted(times, point[3], side="right"), 1, len(times) - 1))
        lower = upper - 1
        weight = (point[3] - times[lower]) / (times[upper] - times[lower])
        if weight == 0:
            return self._slice(lower)(point[:3])
        if weight == 1:
            return self._slice(upper)(point[:3])
        return (1 - weight) * self._slice(lower)(point[:3]) + weight * self._slice(upper)(point[:3])


def _disk_gradient(
    path: Path, directory: Path, axes: tuple[np.ndarray, ...], source_sha256: str
) -> np.ndarray:
    """Create or verify an uncompressed, memory-mapped derivative for repeated rollouts."""
    import h5py

    array_path = directory / "gradient.npy"
    metadata_path = directory / "metadata.json"
    shape = tuple(len(axis) for axis in axes)[::-1]
    if directory.exists():
        if not (metadata_path.exists() and array_path.exists()):
            raise ValueError("Incomplete gradient cache; retain it and choose a new cache directory")
        metadata = json.loads(metadata_path.read_text())
        if metadata["policy_sha256"] != source_sha256:
            raise ValueError("Gradient cache belongs to a different HJB policy")
        with array_path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != metadata["gradient_sha256"]:
            raise ValueError("Gradient cache checksum does not match")
    else:
        directory.mkdir(parents=True)
        partial = directory / "gradient.partial.npy"
        with h5py.File(path) as source, partial.open("wb") as target:
            np.lib.format.write_array_header_2_0(
                target,
                {
                    "descr": np.lib.format.dtype_to_descr(np.dtype(float)),
                    "fortran_order": False,
                    "shape": shape,
                },
            )
            for index in range(shape[0]):
                value = np.asarray(source["value"][index], dtype=float).T
                derivative = np.gradient(value, axes[2], axis=2, edge_order=1)
                derivative.T.tofile(target)
        partial.rename(array_path)
        with array_path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        metadata = {
            "policy_sha256": source_sha256,
            "gradient_sha256": digest,
            "shape": list(shape),
            "dtype": "float64",
            "numpy_version": np.__version__,
            "method": "numpy.gradient along human torque, edge_order=1",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    derivatives = np.load(array_path, mmap_mode="r")
    if derivatives.shape != shape or derivatives.dtype != np.dtype(float):
        raise ValueError("Gradient cache dimensions or dtype do not match the HJB policy")
    return derivatives


class HJBPolicy:
    """Time-dependent trilinear HJB policy with explicit SI axes and strict coverage.

    MAT v7: theta_rad, velocity_rad_s, human_torque_nm, time_to_go_s, value.
    value shape is (Ntheta, Nvelocity, Ntorque, Ntime). Axes must increase.
    NPZ uses the same keys. Export the entire time history, not just its last slice.
    MAT v7.3 validates slices sequentially and caches only two gradient slices.
    """

    def __init__(self, path: str | Path, gradient_cache: str | Path | None = None) -> None:
        path = Path(path)
        hdf5 = False
        if path.suffix == ".npz":
            with np.load(path) as loaded:
                arrays = dict(loaded)
        else:
            import h5py

            if h5py.is_hdf5(path):
                hdf5 = True
                with h5py.File(path, "r") as loaded:
                    arrays = {
                        key: np.asarray(loaded[key]).T
                        for key in ["theta_rad", "velocity_rad_s", "human_torque_nm", "time_to_go_s"]
                    }
                    if "contract_json" in loaded:
                        arrays["contract_json"] = "".join(
                            chr(int(c)) for c in loaded["contract_json"][:].ravel()
                        )
            else:
                arrays = loadmat(path)
        names = ["theta_rad", "velocity_rad_s", "human_torque_nm", "time_to_go_s"]
        axes = [np.asarray(arrays[name], dtype=float).reshape(-1) for name in names]
        if any(len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0) for axis in axes):
            raise ValueError("HJB axes must have at least two finite, increasing SI coordinates")
        shape = tuple(map(len, axes))
        if hdf5:
            with h5py.File(path, "r") as loaded:
                value = loaded["value"]
                if value.shape != shape[::-1] or any(
                    not np.isfinite(value[index]).all() for index in range(shape[3])
                ):
                    raise ValueError("Value function must be finite and match all four grid axes")
            self.gradient = _HDF5Gradient(path.resolve(), tuple(axes))
        else:
            value = np.asarray(arrays["value"], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Value function must be finite and match all four grid axes")
            derivative = np.gradient(value, axes[2], axis=2, edge_order=1)
            self.gradient = RegularGridInterpolator(tuple(axes), derivative, bounds_error=True)
        self.horizon_s = float(axes[3][-1])
        self.minimum_horizon_s = float(axes[3][0])
        self.path = path.resolve()
        self.contract = (
            json.loads(str(np.asarray(arrays["contract_json"]).item())) if "contract_json" in arrays else None
        )
        if gradient_cache is not None:
            if not hdf5:
                raise ValueError("Disk gradient caching requires a v7.3/HDF5 policy")
            derivatives = _disk_gradient(self.path, Path(gradient_cache), tuple(axes), self.sha256)
            self.gradient = _HDF5Gradient(self.path, tuple(axes), derivatives)

    @cached_property
    def sha256(self) -> str:
        """Hash the immutable policy in bounded chunks, once per loaded policy."""
        with self.path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()

    def close(self) -> None:
        """Legacy arrays and optional memory maps follow their normal object lifetime."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def validate_plant(self, profile: Profile, exo: str) -> None:
        """Reject a rollout unless its physical parameters match the exported contract."""
        expected = policy_contract(profile, exo)
        if self.contract is None:
            raise ValueError("HJB rollout requires contract_json from export_hjb_policy")
        if (
            set(self.contract) != set(expected)
            or self.contract["model"] != "paper"
            or self.contract["exo"] != exo
        ):
            raise ValueError("HJB model or exoskeleton does not match the rollout")
        actual = self.contract["profile"]
        if set(actual) != set(expected["profile"]):
            raise ValueError("HJB physical parameter fields are incomplete")
        for key, value in expected["profile"].items():
            if not np.isclose(actual[key], value, rtol=1e-12, atol=1e-12):
                raise ValueError(
                    f"HJB parameter {key} does not match the rollout; select matching --parameters"
                )

    def rate(
        self,
        remaining_s: float,
        lean_rad: float,
        velocity_rad_s: float,
        human_torque_nm: float,
        profile: Profile,
    ) -> float:
        """Return optimal rate, choosing u_min when dV/dtau>0, u_max when <0.

        At zero gradient all controls minimize the Hamiltonian; choose zero rate.
        Out-of-grid states and horizons raise rather than fabricate a policy.
        """
        for endpoint in [self.minimum_horizon_s, self.horizon_s]:
            if abs(remaining_s - endpoint) <= TIME_TOLERANCE_S:
                remaining_s = endpoint
                break
        point = [np.pi / 2 + lean_rad, velocity_rad_s, human_torque_nm, remaining_s]
        if any(
            not axis[0] <= coordinate <= axis[-1]
            for coordinate, axis in zip(point, self.gradient.grid, strict=True)
        ):
            raise PolicyCoverageError(point, self.gradient.grid)
        derivative = float(self.gradient(point).item())
        if derivative > 0:
            return profile.rate_min_nm_s
        if derivative < 0:
            return profile.rate_max_nm_s
        return 0.0


def load_policy(path: str | Path) -> HJBPolicy:
    """Open a legacy value history or a native moving-grid transport policy."""
    import h5py

    from exobalancelab.transport_policy import FORMAT, TransportPolicy

    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as source:
            transport = source.attrs.get("format") == FORMAT
        if transport:
            return TransportPolicy(path)
    return HJBPolicy(path)
