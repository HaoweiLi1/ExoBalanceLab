"""MATLAB controller transport; simulation seconds, joint radians, torque Nm or rate Nm/s.

Replay samples at the current control time (no look-ahead). A single MATLAB Engine
session serves an entire feedback run; startup cost is excluded from call latency.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat


@dataclass
class Trajectory:
    """Validated N-by-C samples; channels follow the receiving actuator order."""

    time_s: np.ndarray
    values: np.ndarray
    interpolation: str = "linear"
    outside: str = "error"

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float).reshape(-1)
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        if (
            self.time_s.size < 2
            or self.values.ndim != 2
            or self.values.shape[0] != self.time_s.size
            or self.values.shape[1] < 1
        ):
            raise ValueError("Trajectory requires at least two times and N-by-C values")
        if not np.isfinite(self.time_s).all() or not np.isfinite(self.values).all():
            raise ValueError("Trajectory contains non-finite samples")
        if np.any(np.diff(self.time_s) <= 0) or self.time_s[0] < 0:
            raise ValueError("Times must be nonnegative and strictly increasing")
        if self.interpolation not in {"linear", "zoh"} or self.outside not in {"error", "hold"}:
            raise ValueError("Use interpolation linear/zoh and outside error/hold")

    @classmethod
    def load(cls, path: str | Path, interpolation: str = "linear", outside: str = "error") -> "Trajectory":
        """Load CSV [time_s, channels...] or MAT variables time_s and torque_nm.

        CSV can have one text header or be the historical two-column numeric format.
        MAT v5-v7.2 and v7.3 numeric arrays are supported. Torque has N rows.
        """
        path = Path(path)
        if path.suffix.lower() == ".mat":
            import h5py

            if h5py.is_hdf5(path):
                with h5py.File(path, "r") as source:
                    times = np.asarray(source["time_s"]).T
                    values = np.asarray(source["torque_nm"]).T
            else:
                source = loadmat(path)
                times, values = source["time_s"], source["torque_nm"]
            # MATLAB vectors may be saved as row or column arrays.
            if values.ndim == 2 and 1 in values.shape:
                values = values.reshape(-1)
        else:
            first = path.read_text().splitlines()[0].split(",")
            try:
                [float(item) for item in first]
                skiprows = 0
            except ValueError:
                skiprows = 1
            data = np.loadtxt(path, delimiter=",", skiprows=skiprows, ndmin=2)
            if data.shape[1] < 2:
                raise ValueError("CSV needs time_s and at least one torque column")
            times, values = data[:, 0], data[:, 1:]
        return cls(times, values, interpolation, outside)

    def sample(self, time_s: float) -> np.ndarray:
        """Interpolate C channels in O(log N), rejecting uncovered time by default."""
        if not np.isfinite(time_s):
            raise ValueError("Sample time must be finite")
        if self.outside == "error" and not self.time_s[0] - 1e-10 <= time_s <= self.time_s[-1] + 1e-10:
            raise ValueError(f"Time {time_s:g} is outside trajectory coverage")
        time_s = np.clip(time_s, self.time_s[0], self.time_s[-1])
        i = int(np.clip(np.searchsorted(self.time_s, time_s, side="right") - 1, 0, len(self.time_s) - 2))
        if self.interpolation == "zoh":
            return self.values[-1 if time_s == self.time_s[-1] else i].copy()
        alpha = (time_s - self.time_s[i]) / (self.time_s[i + 1] - self.time_s[i])
        return self.values[i] + alpha * (self.values[i + 1] - self.values[i])


class MatlabController:
    """Persistent MATLAB function adapter: torque = f(time_s, state_column).

    State schema is defined by the caller, not inferred. The standard paper runner
    sends [lean_rad, velocity_rad_s, human_torque_nm]. Requires licensed MATLAB and
    its release-matched Engine package. Use as a context manager to close the engine.
    """

    def __init__(self, function: str, directory: str | Path, channels: int = 1) -> None:
        if not function.isidentifier() or channels < 1:
            raise ValueError("Expected a MATLAB function identifier and positive channel count")
        self.function = function
        self.directory = Path(directory).resolve(strict=True)
        self.channels = channels
        self.latency_s: list[float] = []
        self.engine = None

    def __enter__(self) -> "MatlabController":
        import matlab.engine

        self.engine = matlab.engine.start_matlab()
        try:
            self.engine.addpath(str(self.directory), nargout=0)
        except Exception:
            self.engine.quit()
            self.engine = None
            raise
        return self

    def sample(self, time_s: float, state: np.ndarray) -> np.ndarray:
        """Evaluate one feedback command and record transport plus evaluation time."""
        import matlab

        if self.engine is None:
            raise RuntimeError("Open MatlabController with a context manager first")
        start = perf_counter()
        result = self.engine.feval(
            self.function, float(time_s), matlab.double(np.asarray(state).reshape(-1, 1).tolist()), nargout=1
        )
        self.latency_s.append(perf_counter() - start)
        torque = np.asarray(result, dtype=float).reshape(-1)
        if torque.size != self.channels or not np.isfinite(torque).all():
            raise ValueError("MATLAB output must be finite and match the actuator channel count")
        return torque

    def __exit__(self, *exc) -> None:
        if self.engine is not None:
            self.engine.quit()
            self.engine = None
