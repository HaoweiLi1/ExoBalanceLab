# MATLAB integration

## Offline controller exchange

`export_controller(path,time_s,torque_nm)` writes CSV or MAT v7 samples. Time is N-by-1,
strictly increasing, nonnegative seconds; torque is N-by-C human torque in Nm. `example_export.m`
creates a simple six-second signal. No license is needed to consume these exports in Python.
The reader also supports numeric MATLAB v7.3/HDF5 arrays, handles MATLAB row/column vectors,
and transposes HDF5 storage dimensions back into MATLAB logical order.

- `paper` and `contact`: C=1, human ankle torque.
- `human`: C=6, left hip/knee/ankle, then right hip/knee/ankle.
- Headerless CSV remains compatible with `joint1_humanoid/pre_traj/ankle_torque_trajectory.csv`.
- Linear interpolation is the default; `--interpolation zoh` preserves discrete commands.
- Missing files, NaN, duplicate times, wrong channel counts and uncovered horizons fail explicitly.
- The Python API permits `outside="hold"` only when the caller explicitly requests endpoint hold.
- The requested human torque is rate/torque limited; raw and applied commands are saved.
- Do not export *total* human+exo torque as human torque or assistance will be counted twice.

## Persistent MATLAB feedback

Install the Engine supplied with the installed MATLAB release into the Conda environment:

```bash
python -m pip install /path/to/MATLAB/extern/engines/python
exobalance run --matlab-function standing_pd --matlab-directory matlab \
  --control-period .01 --duration 6 --out logs/matlab_feedback
```

`standing_pd(time_s,state)` receives a MATLAB column vector `[q;qdot;human_torque]` and
returns one desired human torque. It is a no-assistance OF example. The human-model
state is `[qpos(9);qvel(9);applied_human_torque(6)]`, returning six torques.

One MATLAB session starts per episode, not per physics step. It closes on success or
exception. Function output must match the channel count and contain finite numbers.
`latency_s` records each Engine call. The runner also records control-call p50/p99.
Simulation advances in simulated time; it does not guarantee wall-clock deadlines.
Controllers with persistent state start fresh for every paired run.

### Linux runtime compatibility

On the development machine (Ubuntu 22.04, MATLAB R2026b), importing NumPy/MuJoCo
before the Engine loaded the system C++ runtime, which lacks `GLIBCXX_3.4.31`.
Selecting MATLAB's supplied runtime for this process resolves the import error:

```bash
MATLAB_ROOT=/path/to/MATLAB/R2026b
LD_LIBRARY_PATH="$MATLAB_ROOT/sys/os/glnxa64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  python -c 'import numpy, mujoco, matlab.engine; print("Engine import OK")'

LD_LIBRARY_PATH="$MATLAB_ROOT/sys/os/glnxa64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  EXOBALANCE_MATLAB_TEST=1 python -m pytest -q -s tests/test_matlab_runtime.py
```

Use the same process-local prefix for `exobalance run --matlab-function ...` if
affected. No system library replacement is needed. Import success checks the binary
dependencies only; the test still needs MATLAB to start with a valid license.
Account activation and an authenticated MATLAB runtime are separate checks: verify
that the desktop reaches its `>>` command prompt and that `matlab -batch "disp(version)"`
exits successfully before running unattended experiments.

## HJB torque-rate feedback

The paper controls **human torque rate**, with human torque as the third state.
After a matching helperOC solve, export the complete value history:

```matlab
export_hjb_policy('policy.mat', g, data, tau, dPend);
```

The exporter removes the author's nondimensional velocity, torque and time scales.
Python loads `theta_rad`, `velocity_rad_s`, `human_torque_nm`, `time_to_go_s`, and
`value[Ntheta,Nvelocity,Ntorque,Ntime]`. It interpolates the torque derivative of the
value function and selects the minimizing rate bound from Eq. (11). Out-of-grid queries
raise; an unrelated controller is never used as a fallback.
The selected rate is held between control ticks, and human torque is integrated at
each physics step from the configured initial torque. It is not applied as a torque
jump at a control tick. Torque saturation and commanded/applied rates are recorded.

```bash
exobalance run --hjb-policy policy.mat --profile OF --exo none \
  --duration 1 --out logs/hjb_feedback
```

The policy must be generated for the same profile, physical parameters, exoskeleton law,
constraint set and time horizon as the MuJoCo experiment. The exporter does **not** repair
inconsistencies in the upstream solver. In particular, the source gravity value, OF/YF
heights, GC implementation and missing dependencies differ from the paper-linked default;
see [the source audit](../docs/PAPER.md). Do not interpret an arbitrary exported policy
as an exact reproduction merely because the file loads.

## Validation status

Transport, interpolation, sampling, torque-rate sign and closed-loop MuJoCo execution are
covered by automated tests. HJB execution tests check initial torque, rate hold, saturation
and timestep convergence against an independently integrated three-state ODE.
Synthetic HJB fixtures test execution only, not reachability or the published regions.
The development machine has MATLAB R2026b, its matching Engine and Optimization Toolbox.
Following reported activation, batch startup still requested a MathWorks sign-in and
exited with `Sign-in failed`; the R2026b desktop also displayed a sign-in dialog.
NumPy, MuJoCo and the Engine import together successfully with the Linux runtime prefix
above. MATLAB-generated round-trip, closed-loop parity and real Engine latency remain
unverified until runtime sign-in succeeds.

## Prepared HJB dependencies

The following sources were downloaded unchanged on October 22, 2026, under ignored
`logs/hjb_dependencies/`. These are preparation records, not MATLAB solver test results.
Keep vendor licenses with each download and apply any future source corrections separately.

| Dependency | Source | Pinned version |
|---|---|---|
| helperOC | [HJReachability/helperOC](https://github.com/HJReachability/helperOC) | `4ff68174c616d8aa4dccbee9ebf36665999c90f1` |
| ToolboxLS | [Ian Mitchell's release archive](https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.1.zip) | `1.1.1`; SHA-256 below |
| YALMIP | [yalmip/YALMIP](https://github.com/yalmip/YALMIP) | `52666d27c40e5eca5ed72de713f05e5959daf937` |
| SeDuMi | [sqlp/sedumi](https://github.com/sqlp/sedumi), maintained fork | `daef0b7ee87c727a6d97bc1ca892f48b599166d7` |

ToolboxLS archive SHA-256:
`8ac634e3b91a1d27827c156880428b35bd08d951580dd9e80fa064e84f5849e8`.
Use the ToolboxLS `Kernel` directory and helperOC's documented setup paths. SeDuMi's
source checkout needs `install_sedumi` and working MEX binaries or a compatible compiler.
The authors also require Matt J's
[Analyze N-dimensional Convex Polyhedra](https://www.mathworks.com/matlabcentral/fileexchange/30892-analyze-n-dimensional-convex-polyhedra)
for `lcon2vert`/`vert2lcon`; that download and runtime validation are still pending.
The missing friction function and target/controller discrepancies in
[the source audit](../docs/PAPER.md) must be resolved before computing comparable regions.
