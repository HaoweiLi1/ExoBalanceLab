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
covered by automated tests. Synthetic HJB fixtures test the interface only, not reachability.
The development machine has MATLAB R2026b and its matching Engine; activation is currently
required before MATLAB execution can be verified. MATLAB-generated round-trip and real
Engine benchmarks remain pending until that activation succeeds.
