# Python API and model contracts

Install from the repository root with `pip install -e '.[dev]'`. Imports start no simulator,
GUI or MATLAB process. Heavy rendering imports happen only when requested.

## Public interfaces

| Module | Interface | Inputs → outputs |
|---|---|---|
| `bridge` | `Trajectory.load(path, interpolation, outside)` | CSV/MAT → validated `Trajectory` |
| `bridge` | `Trajectory.sample(time_s)` | seconds → C desired torques (Nm) |
| `bridge` | `MatlabController(function, directory, channels)` | context manager owning one Engine |
| `bridge` | `MatlabController.sample(time_s, state)` | seconds/SI column state → torque vector |
| `paper` | `Profile`, `PROFILES`, `AUTHORS_PROFILES`, `PROFILE_SETS` | Printed/source anthropometry, gravity, MT and MRTD |
| `paper` | `exo_torque(q, velocity, profile, mode)` | rad/rad/s → bilateral Nm |
| `paper` | `acceleration(q, velocity, total_torque, profile)` | Eq. (3) → rad/s² |
| `paper` | `ground_reaction(q, velocity, total_torque, profile)` | flat-foot inverse dynamics → N/metres |
| `paper` | `limit_human(raw, previous, dt, profile)` | desired/applied torque, s → limited Nm |
| `models` | `paper_xml(profile, dt, contact)` | profile → standalone MJCF string |
| `models` | `human_xml(profile, dt, device_mass_kg)` | profile → bilateral sagittal MJCF |
| `models` | `compile_model(xml)` | MJCF → MuJoCo model/data |
| `simulation` | `RunConfig`, `RunConfig.validate()` | explicit SI episode configuration |
| `simulation` | `run(config, trajectory, matlab, hjb)` | at most one external policy → `Episode` |
| `simulation` | `Episode.save(directory)` | records → new evidence directory |
| `reachability` | `HJBPolicy(path, gradient_cache=None)`, `HJBPolicy.rate(...)` | SI value grid → human torque rate Nm/s; optional disk derivative cache |
| `reachability` | `policy_contract(profile,exo)`, `HJBPolicy.validate_plant(profile,exo)` | Exported physical parameters → checked rollout compatibility |
| `reachability` | `PolicyCoverageError.query`, `.bounds` | Failed query and exported bounds in SI units |
| `rendering` | `comparison_video(episodes, output, fps, width, height, titles)` | synchronized records and optional panel labels → MP4/GIF/poster |
| `rendering` | `plot_comparison(episodes, output)` | records → PNG time series |
| `validation` | `validate_dynamics(seed)`, `benchmark_bridge(seed)` | deterministic protocol → metrics |
| `hjb_validation` | `validate(run_dir, output, count, seed, protocol_path, timestep_s, control_period_s, gradient_cache)` | Matching nominal/assisted HJB → new or frozen-cohort rollouts with explicit timing |
| `hjb_comparison` | `compare(runs, output)` | Matching cohort/plant/targets → complete per-case CSV, JSON, PNG and PDF |
| `hjb_regions` | `report(protocol_path, output)` | Complete 30-case campaign → projected areas, Table II comparison and figures |
| `hjb_regions` | `compare_grids(protocol_path, output)` | Declared one-profile grid study → native/common-grid area and boundary comparisons |

`cli`, `sweep`, `validation`, `hjb_validation`, `hjb_comparison` and `hjb_regions` expose
executable `main()` functions.

Grid-study protocols require all six assistance/direction pairs by default. A targeted
follow-up can explicitly list `mode_direction_pairs`; every listed pair must be present
at every declared grid resolution.

## Timing and signs

`mj_forward` computes the observation at time t after setting the current command;
`mj_step` advances to t+dt. Recorded states/commands share the pre-step timestamp.
Desired human torque controllers (replay, MATLAB and local feedback) update every
`control_period_s`, with zero-order hold between updates. Their MRTD limiter permits
one control interval of change at t=0. Exoskeleton commands use that same period.
Physics dt must divide the control period and duration exactly; timing uses simulated
time, not wall-clock call latency. No future-time lookup compensates for transport delay.

HJB controls **torque rate**, so its lifted torque state starts exactly at
`initial_torque_nm`. The selected rate is held between controller ticks, while torque
is integrated every physics step and clipped to the strength-scaled MT limits. The
physics step uses the current torque before advancing this state. This first-order
discretization converges to the continuous lifted dynamics as physics dt decreases.
HJB records additionally contain `human_rate_command_nm_s` and
`human_rate_applied_nm_s`; the latter is the torque change over the following physics
interval divided by dt, including saturation. All initial torques must satisfy the
configured strength-scaled limits.
Episode metrics `final_lean_rad`, `final_velocity_rad_s` and `final_human_torque_nm`
describe the actual post-step endpoint; HJB terminal-target checks use that state.

`RunConfig.parameters` / `--parameters` selects `table_i` (default printed Table I,
g=9.81) or `authors` (pinned authors' MAT anthropometry, g=9.8). The source has OF
height 1.58 m and YF height 1.653 m; other profiles retain their printed heights.
The effective profile is recorded in episode metrics, and gravity is set in the MJCF.
HJB files must include `contract_json`; `run` validates it against the effective profile
and assistance mode before constructing the plant. This catches parameter/strength
mismatches, but does not prove that a value function solves the declared HJB equation.
MAT v7.3 policies use two cached gradient slices; validation and SHA-256 hashing also
stream from disk. This preserves four-dimensional linear interpolation without holding
the full value history and its derivative in memory.
For repeated rollouts, `HJBPolicy(path, gradient_cache=directory)` optionally stores an
uncompressed derivative on disk and reads it through a memory map. Source and cache
checksums prevent reuse with another policy; the interpolation arithmetic is unchanged.

For `paper`/`contact`, positive ankle angle is forward lean; positive motor torque is DF.
A body hinge about world +y produces positive forward lean. In `human`, the ankle joint
rotates the foot relative to the shank, so its axis is -y: with flat feet its angle and
reaction torque on the shank have the same signs as the paper. Proximal pitch joints
use +y. Units are radians and Nm throughout; angle displays convert to degrees.

Human qpos/qvel order:

1. root x (m / m/s), root z (m / m/s), root pitch (rad / rad/s)
2. left hip, left knee, left ankle
3. right hip, right knee, right ankle

Six human torque channels: left hip/knee/ankle, right hip/knee/ankle. Two additional
MuJoCo controls are left/right device ankle torque. Knee/hip limits are engineering
assumptions (±100 Nm, ±500 Nm/s); ankle limits are half the bilateral Table I values.
The torso, head and arms have distributed mass but move rigidly with the pelvis.
The device has 70% of its mass on the two shanks and 30% on the two feet.

## Controllers and recovery

The reduced model uses an exoskeleton-aware human PD target, then MT/MRTD limiting.
The articulated model linearizes the actual settled contact plant and solves a discrete
LQR for six human torques. The same gain and initial state are recomputed deterministically
for each assistance mode. The human controller is synthetic full-state feedback, not a
fitted physiological model. Device mass stays attached when assistance is off.

`recovered` requires the final min(1 s, duration) to satisfy |lean|<0.05 rad and
|lean velocity|<0.05 rad/s, plus valid contact over the whole recorded episode.
Contact requires positive normal load, in-foot CoP and friction margin, plus foot
pitch <0.035 rad and foot translation <0.02 m. These last two tolerances account for
compliant MuJoCo contact and prevent a tipped/sliding foot from being called feet-in-place
recovery. For `human`, COM height must also remain above 60% of its initial value.
Unassisted settling precedes contact-model recordings and is excluded from the experiment clock.

`normal_n`, `friction_n`, `cop_m` are **required analytical** reactions in `paper` and
**measured floor contacts** in the other models. There is no physically measured contact
force in a fixed-foot model. `human_torque_nm` sums ankle channels only; the full raw/applied
six-channel arrays remain in the NPZ. RMS torque metrics aggregate all human actuators,
so they are not estimates of metabolic cost.

Use an existing episode's XML and arrays for offline inspection. Renderer frames use the
first recorded sample at or after the requested display time (error < physics dt).
Wall-clock performance excludes rendering and model initialization; see the result labels.
