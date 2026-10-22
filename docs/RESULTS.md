# Measured results — October 22, 2026

All results below were obtained with Python 3.11, MuJoCo 3.3.7 and the committed model
configuration files. Complete numerical/video run snapshots are in [results](results/).
The research statement is deliberately narrower than “complete reproduction”: original
HJB regions and real MATLAB runtime execution remain separate pending acceptance.

## Mechanics and MATLAB-file transport

[Validation data](results/validation.json) records seed 240805418 and all profile parameters.

- **1,500 random-state acceleration comparisons**, 300 per Table I profile: maximum
  absolute MuJoCo-versus-independent-Eq. (3) error **1.16e-9 rad/s²**.
- **15 independent ODE integration comparisons**, five profiles and three timesteps:
  halving dt approximately halves state error, as expected for the selected integrator.
  These are 0.5 s constant-torque comparisons; they do not establish long-horizon stability.
- **12,001-sample trajectory, 1,000 off-grid queries:** historical nearest-neighbor linear
  scan 1,801 µs/query versus binary-search interpolation 20.5 µs/query, **87.8×** in this run.
  The continuous-signal error also fell from 2.45e-4 to 3.12e-8 for the sinusoidal fixture.
  This is in-process lookup cost, not a MATLAB Engine or end-to-end real-time benchmark.
- **33 automated tests passed**, with the licensed MATLAB test skipped pending runtime access.
  These include native MuJoCo mechanics, static pressure-center/angle signs, MAT/CSV interchange,
  controller timing, legacy adapters, torque limits and synthetic HJB policy semantics.

Reproduce:

```bash
python -m exobalancelab.validation --out logs/validation
pytest -q
# After successful MATLAB runtime sign-in and matching Engine installation:
# On affected Linux systems, use the runtime prefix in matlab/README.md.
EXOBALANCE_MATLAB_TEST=1 pytest -q -s tests/test_matlab_runtime.py
```

## Assistance can enlarge and shrink a fixed controller's recovery region

![Recovery comparison](results/recovery/recovery.png)

The OF experiment runs all **13×13 initial states × 3 modes = 507 episodes**.
The grid spans q∈[-0.08,0.20] rad and velocity∈[-0.6,0.6] rad/s; initial human torque is zero.
Every run lasts 6 s at 500 Hz physics/control. A final one-second standing interval and
C3–C5 validity for the full trajectory are required. The complete
[protocol](results/recovery/protocol.json), [per-episode CSV](results/recovery/grid.csv) and
[summary](results/recovery/summary.json) retain failures as well as successes.

| Mode | Recovered / 169 | Newly recovered vs none | Lost recovery vs none |
|---|---:|---:|---:|
| No assistance | 35 | — | — |
| GC | 32 | 9 | 12 |
| PD | 42 | 11 | 4 |

This illustrates the competing mechanisms highlighted by the paper. It is not a
reproduction of its Table II: a fixed feedback law, finite grid, six-second horizon and
zero-torque slice differ from a maximal HJB region projected over initial human torque.

```bash
python -m exobalancelab.sweep --points 13 --out logs/recovery_grid
```

## Articulated human: inspect the device's effect

![Human comparison](media/human.gif)

[MP4](media/human.mp4) · [Torque/lean curves](media/human_signals.png) ·
[Full run records](results/human/)

OF anthropometry; 60 kg human plus 2 kg bilateral device; initial upright rest; the same
**80 N anterior chest force from 1.00 to 1.15 s** in each run; 1 kHz physics/control; 6 s total.
The physical device mass is retained in the motor-off control condition. One synthetic
human LQR, unchanged across modes, controls the six hip/knee/ankle actuators.

- Motor off: the model falls and loses the feet-in-place constraints.
- GC and PD: both return to standing with valid contact throughout the episode.
- These outcomes concern this model and controller, not clinical fall prevention.

The 80 N demonstration was selected after exploratory runs. A broader fixed force list
is reported in full in [the 33-run force sweep](results/human_force_sweep/summary.json),
with its [protocol](results/human_force_sweep/protocol.json). Do not interpret a selected
video as a random trial or a population-level success rate. The force sweep also contains
conditions where assistance fails.

```bash
MUJOCO_GL=egl exobalance compare --model human --lean 0 --velocity 0 \
  --push 80 --duration 6 --video --out logs/human_demo
```

## Constraint failure is not always visible as a fall

[Reduced-model video](media/paper_constraints.mp4) ·
[Curves](media/paper_constraints_signals.png) · [Records](results/paper_constraints/)

The reduced model starts at q=-0.08 rad, velocity=+0.2 rad/s. Without assistance it reaches
standing while satisfying required foot reactions. With GC or PD, required contact
constraints are briefly violated even though the *fixed-foot* pendulum eventually becomes
upright. The video labels these as required reactions. The constrained model cannot show
physical foot tipping; it does not prove that the person has recovered safely.

This example was selected from the reported grid. It is not equivalent to the articulated
chest-push experiment. The `contact` extension permits the foot to move; its compliant
contacts and finite body COM inertia can yield different outcomes. Do not silently replace
one model's recovery classification with another's.

```bash
MUJOCO_GL=egl exobalance compare --model paper --lean -.08 --velocity .2 \
  --duration 6 --video --out logs/paper_constraints
```

## Provenance and remaining acceptance

Each published episode includes raw/applied torques, time-aligned states, model XML,
configuration, numerical metrics and source/model hashes. These captures were generated
from the implementation working tree before the documentation/media commit; the recorded
Git SHA names the base commit and `working_tree_dirty` makes that distinction explicit.
Model XML and full arrays make the rendered evidence directly inspectable.

Earlier development runs used a superseded forward/backward convention and are retained
only under ignored local `logs/`. They are not included in the published results above.

Pending: finish runtime sign-in for installed MATLAB R2026b, execute the licensed round-trip/feedback
test, reconcile the upstream source discrepancies, and reproduce the original HJB regions
with convergence/area comparisons. See [paper acceptance](PAPER.md). The articulated model
also remains unvalidated against subject-specific mechanics or physical human experiments.
