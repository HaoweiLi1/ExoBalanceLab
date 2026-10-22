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
Time endpoints allow at most 1e-12 s of SI-conversion roundoff; physical grid bounds
remain strict. This handles an exported 1.3999999999999997 s endpoint for a 1.4 s run.
The selected rate is held between control ticks, and human torque is integrated at
each physics step from the configured initial torque. It is not applied as a torque
jump at a control tick. Torque saturation and commanded/applied rates are recorded.
Rollouts also require embedded `contract_json` identifying the paper model, assistance
mode, mass, height, gravity, damping, friction and effective MT/MRTD limits. The exporter
derives this automatically for the authors' unchanged nominal `LiftedInvPendExo` class.
Other callers must supply a sixth `contract` struct; its schema matches Python's
`policy_contract(profile, exo)` dictionary. Unbound files can be inspected with
`HJBPolicy.rate`, but cannot drive a rollout. Mismatched plants fail before simulation.

```bash
exobalance run --hjb-policy policy.mat --profile OF --parameters authors --exo none \
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
Runtime sign-in, CSV/MAT v7 round trips, a real v7.3 HJB fixture export and closed-loop
MATLAB/Python parity now pass. A six-second, 600-call run measured median Engine call
latency 1.790 ms and p99 2.927 ms, with three calls exceeding the 10 ms control period.
See [all measurements and retained episodes](../docs/RESULTS.md#real-matlab-feedback-and-exports).
The Linux runtime prefix above is needed on the development machine.

## Prepared HJB dependencies

The following sources were downloaded on October 22, 2026, under ignored
`logs/hjb_dependencies/`. SeDuMi's 34 MEX modules were compiled with MATLAB's selected
GCC compiler. YALMIP/SeDuMi passed a positive-semidefinite optimization check, and
the polyhedra utility recovered the eight vertices of a cube in real MATLAB.
Keep vendor licenses with each download and apply any future source corrections separately.

| Dependency | Source | Pinned version |
|---|---|---|
| helperOC | [HJReachability/helperOC](https://github.com/HJReachability/helperOC) | `4ff68174c616d8aa4dccbee9ebf36665999c90f1` |
| ToolboxLS | [Ian Mitchell's release archive](https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.1.zip) | `1.1.1`; SHA-256 below |
| YALMIP | [yalmip/YALMIP](https://github.com/yalmip/YALMIP) | `52666d27c40e5eca5ed72de713f05e5959daf937` |
| SeDuMi | [sqlp/sedumi](https://github.com/sqlp/sedumi), maintained fork | `daef0b7ee87c727a6d97bc1ca892f48b599166d7` |
| Polyhedra | [Matt J's File Exchange submission](https://www.mathworks.com/matlabcentral/fileexchange/30892-analyze-n-dimensional-convex-polyhedra) | Submission 30892, download version 6; SHA-256 below |

ToolboxLS archive SHA-256:
`8ac634e3b91a1d27827c156880428b35bd08d951580dd9e80fa064e84f5849e8`.
Polyhedra [download archive](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/30892/versions/6/download/zip)
SHA-256: `8edb364aeedb2f5d80c7a43ffe58845a6875179c9e7febd144c68f1c35f55389`.
Keep its BSD license with `lcon2vert.m` and `vert2lcon.m`. Use the ToolboxLS `Kernel`
directory and helperOC's documented setup paths. SeDuMi's source checkout needs
`install_sedumi` and working MEX binaries or a compatible compiler.

## Nominal HJB baseline

`solve_nominal_hjb(author_root, dependency_root, output_dir, ...)` provides an executable
no-exoskeleton baseline using the authors' stored model/target MAT files and unchanged
dynamics. Download the pinned authors' repository and dependencies above first; unpack
ToolboxLS as `dependency_root/ToolboxLS/ToolboxLS/Kernel`.

```matlab
addpath('matlab');
solve_nominal_hjb('/path/to/AnkleExoBRS', 'logs/hjb_dependencies', 'logs/hjb_OF_forward', ...
    profile="OF", direction="forward", resolution=[51;225;51]);
```

Options are `profile` (five paper labels), `direction` (`forward` or `backward`),
`resolution` (three grid sizes), `duration_s` (default 1.4), `output_step_s` (default 0.1),
`velocity_buffer_rad_s` (default 0.02), `quiet` (default false),
and `method` (`none` or `minVWithV0`, default `none`). This coarse default is for bringing
up the solver, not demonstrating spatial convergence. MATLAB's `-singleCompThread`
startup option limits CPU use but was substantially slower on the development machine.
Use default computational threads for the measured run. Progress is printed unless
`quiet=true`; a MATLAB `-logfile` preserves it without requiring an interactive window.

The runner stages the original class files into an `@LiftedInvPendExo` directory inside
the new run directory so MATLAB can find its external methods. Original checkouts and
previous runs are preserved. Empty sampled targets raise an error rather than being enlarged.
Outputs include `policy.mat`, final `projection.mat`, input MAT snapshots, staged class
source and `summary.json`. Exports retain the complete sampled value history in SI units.

`nominal_contact_constraints(x,pendulum)` reconstructs force/moment constraints directly
from the nominal dynamics, filling the missing upstream friction implementation. It uses
positive admissibility margins, friction coefficient 1, human torque bounds, and moment
inequalities for CoP without dividing by normal force. Both SI and the authors' scaled
states are supported; exoskeleton dynamics are explicitly rejected. The constraints passed
500 random-state comparisons with MuJoCo under all five Table I parameter sets.

The actual solver uses the **authors' parameters**, including g=9.8 and OF height 1.58 m;
select `--parameters authors` for matching MuJoCo experiments. The default `table_i`
parameter set retains the printed anthropometry and g=9.81. Embedded contracts prevent
mixing these plants. GC/PD consistency, target-set invariance, convergence, matching
forward rollouts over the full regions and published-area comparison remain
open. The [source audit](../docs/PAPER.md) describes the differences, including the solver
method argument; a successful nominal solve does not reproduce all published regions.
The box targets follow the numerical approximation explicitly described in Appendix II-B;
they do not inherit a proof of invariance from their enclosed CLF ellipsoids. The default
opposite-direction velocity buffer is the source's 0.02 rad/s, whereas Appendix III
reports 0.5 rad/s. Keep this domain difference explicit when comparing published results.

After a solve, select and record inside/outside/boundary states before viewing any
forward outcomes, then test the exported policy in the matching MuJoCo plant:

```bash
python -m exobalancelab.hjb_validation --run-dir logs/hjb_OF_forward \
  --out logs/hjb_OF_forward_validation --count 3 --seed 240805418
```

The protocol records the policy hash and all selected states. Outcomes distinguish
target reach, contact validity and explicit rollout errors; an out-of-grid policy query
is retained as an error. The final target test uses the actual end-of-horizon lifted
state. Completed episodes and the projected region plot are saved alongside the report.
Use `--protocol path/to/earlier/protocol.json` to retain exactly the same initial states
and selection groups when comparing grids or time histories. Values and inside/outside
predictions are recomputed for the new policy; `selection_value` retains the earlier value.
The original selection seed, policy hash and protocol hash identify the frozen cohort.
Coverage failures record the query, axis bounds and elapsed simulation time in SI units.
The published 1.4 s coarse-grid run includes prediction/rollout disagreements and
out-of-grid failures; see [all outcomes](../docs/RESULTS.md#longer-horizon-coarse-grid-prediction-is-not-sufficient).
