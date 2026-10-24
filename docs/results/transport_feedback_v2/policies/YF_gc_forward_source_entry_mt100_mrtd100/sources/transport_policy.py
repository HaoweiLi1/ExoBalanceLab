"""Read native transport feedback on moving SI grids, with a fixed terminal-goal choice.

Files store exact three-way Hamiltonian decisions at every numerical time step.
Trilinear state and linear time interpolation produce bounded human torque rates.
The terminal goal is selected from the initial state's final value, then held.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from exobalancelab.paper import Profile
from exobalancelab.reachability import TIME_TOLERANCE_S, HJBPolicy, PolicyCoverageError

FORMAT = "exobalance.transport-feedback.v1"


def switching_codes(value: np.ndarray) -> np.ndarray:
    """Encode the central torque derivative sign: positive selects u_min, negative u_max."""
    codes = np.empty(value.shape, dtype=np.int8)
    # Comparing operands gives the same sign without allocating float64 differences.
    for result, upper, lower in [
        (codes[..., 1:-1], value[..., 2:], value[..., :-2]),
        (codes[..., 0], value[..., 1], value[..., 0]),
        (codes[..., -1], value[..., -1], value[..., -2]),
    ]:
        np.greater(upper, lower, out=result, casting="unsafe")
        np.subtract(result, np.less(upper, lower), out=result)
    return codes


def interpolate_cell(field, axes, point, rates=None):
    """Interpolate the eight enclosing samples, optionally decoding exact rate choices."""
    starts, weights = [], []
    for coordinate, axis in zip(point, axes, strict=True):
        index = int(np.clip(np.searchsorted(axis, coordinate, side="right") - 1, 0, len(axis) - 2))
        starts.append(index)
        weights.append((coordinate - axis[index]) / (axis[index + 1] - axis[index]))
    values = np.asarray(field[tuple(slice(i, i + 2) for i in starts)], dtype=float)
    if rates is not None:
        values = np.where(values > 0, rates[0], np.where(values < 0, rates[1], 0.0))
    for weight in weights:
        values = (1 - weight) * values[0] + weight * values[1]
    return float(values)


class TransportPolicy(HJBPolicy):
    """Own a lossless HDF5 decision table; use as a context manager to close it."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._file = h5py.File(self.path, "r", rdcc_nbytes=32 * 1024**2)
        try:
            if self._file.attrs.get("format") != FORMAT or not self._file.attrs.get("complete", False):
                raise ValueError("Incomplete or incompatible transport feedback export")
            self.contract = json.loads(self._file.attrs["contract_json"])
            self.times = self._file["time_to_go_s"][:]
            self.horizon_s = float(self.times[-1])
            self.minimum_duration_s = self.horizon_s
            self.minimum_horizon_s = float(self.times[0])
            self.rates = np.array(self._file["rate_bounds_nm_s"])
            self.goals = []
            for group in self._file["goals"].values():
                axes = tuple(group[name][:] for name in ["theta_rad", "velocity_rad_s", "human_torque_nm"])
                self.goals.append((group, axes))
            self.selected_goal = None
        except Exception:
            self._file.close()
            raise

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def initial_values(self, state):
        """Return each goal's final value at the initial SI state, using the full native field."""
        values = []
        for group, axes in self.goals:
            grid = tuple(a[-1] for a in axes)
            if any(not a[0] <= x <= a[-1] for x, a in zip(state, grid, strict=True)):
                raise PolicyCoverageError([*state, self.horizon_s], (*grid, self.times))
            values.append(interpolate_cell(group["final_value"], grid, state))
        return values

    def rate(
        self,
        remaining_s: float,
        lean_rad: float,
        velocity_rad_s: float,
        human_torque_nm: float,
        profile: Profile,
    ) -> float:
        state = [np.pi / 2 + lean_rad, velocity_rad_s, human_torque_nm]
        for endpoint in [self.minimum_horizon_s, self.horizon_s]:
            if abs(remaining_s - endpoint) <= TIME_TOLERANCE_S:
                remaining_s = endpoint
        if self.selected_goal is None or remaining_s == self.horizon_s:
            self.selected_goal = int(np.argmin(self.initial_values(state)))
        group, axes = self.goals[self.selected_goal]
        if not self.times[0] <= remaining_s <= self.times[-1]:
            raise PolicyCoverageError([*state, remaining_s], (*(a[-1] for a in axes), self.times))
        upper = int(np.clip(np.searchsorted(self.times, remaining_s, side="right"), 1, len(self.times) - 1))
        lower = upper - 1
        weight = (remaining_s - self.times[lower]) / (self.times[upper] - self.times[lower])
        answer = 0.0
        for index, fraction in [(lower, 1 - weight), (upper, weight)]:
            if fraction == 0:
                continue
            grid = tuple(a[index] for a in axes)
            if any(not a[0] <= x <= a[-1] for x, a in zip(state, grid, strict=True)):
                raise PolicyCoverageError([*state, remaining_s], (*grid, self.times))
            answer += fraction * interpolate_cell(group[f"controls/{index:04d}"], grid, state, self.rates)
        return float(np.clip(answer, profile.rate_min_nm_s, profile.rate_max_nm_s))
