"""Replay exported MATLAB/helperOC HJB feedback on the paper's lifted state.

Coordinates: absolute theta [rad], angular velocity [rad/s], HUMAN torque [Nm],
remaining horizon [s]. Value gradients select human torque RATE [Nm/s], Eq. (11).
A sampled recovery controller is never substituted for missing HJB evidence.
"""

import json
from dataclasses import asdict
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


class HJBPolicy:
    """Time-dependent trilinear HJB policy with explicit SI axes and strict coverage.

    MAT v7: theta_rad, velocity_rad_s, human_torque_nm, time_to_go_s, value.
    value shape is (Ntheta, Nvelocity, Ntorque, Ntime). Axes must increase.
    NPZ uses the same keys. Export the entire time history, not just its last slice.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path) as loaded:
                arrays = dict(loaded)
        else:
            import h5py

            if h5py.is_hdf5(path):
                with h5py.File(path, "r") as loaded:
                    arrays = {
                        key: np.asarray(loaded[key]).T
                        for key in ["theta_rad", "velocity_rad_s", "human_torque_nm", "time_to_go_s", "value"]
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
        value = np.asarray(arrays["value"], dtype=float)
        if value.shape != tuple(map(len, axes)) or not np.isfinite(value).all():
            raise ValueError("Value function must be finite and match all four grid axes")
        derivative = np.gradient(value, axes[2], axis=2, edge_order=1)
        self.gradient = RegularGridInterpolator(tuple(axes), derivative, bounds_error=True)
        self.horizon_s = float(axes[3][-1])
        self.minimum_horizon_s = float(axes[3][0])
        self.path = path.resolve()
        self.contract = (
            json.loads(str(np.asarray(arrays["contract_json"]).item())) if "contract_json" in arrays else None
        )

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
        try:
            derivative = float(self.gradient(point).item())
        except ValueError as error:
            raise PolicyCoverageError(point, self.gradient.grid) from error
        if derivative > 0:
            return profile.rate_min_nm_s
        if derivative < 0:
            return profile.rate_max_nm_s
        return 0.0
