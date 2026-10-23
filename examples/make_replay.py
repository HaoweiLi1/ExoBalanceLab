"""Create equivalent single-channel CSV/MAT torque samples without running MATLAB."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("logs/replay_input"))
    args = parser.parse_args()
    # A small transport example, not an optimized balance controller.
    time_s = np.linspace(0, 1, 101)
    torque_nm = (0.5 * np.sin(2 * np.pi * time_s))[:, None]
    args.out.mkdir(parents=True, exist_ok=False)
    np.savetxt(
        args.out / "torque.csv",
        np.column_stack([time_s, torque_nm]),
        delimiter=",",
        header="time_s,human_ankle_nm",
        comments="",
        fmt="%.17g",
    )
    savemat(args.out / "torque.mat", {"time_s": time_s[:, None], "torque_nm": torque_nm})
    print(f"Wrote equivalent CSV and MAT samples to {args.out.resolve()}")


if __name__ == "__main__":
    main()
