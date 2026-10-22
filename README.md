# ExoBalanceLab

**MATLAB controllers. MuJoCo contact dynamics. Human–exoskeleton balance you can inspect.**

[![Tests](https://github.com/HaoweiLi1/ExoBalanceLab/actions/workflows/tests.yml/badge.svg)](https://github.com/HaoweiLi1/ExoBalanceLab/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](environment.yml)
[![MuJoCo 3.3.7](https://img.shields.io/badge/MuJoCo-3.3.7-orange)](pyproject.toml)
[Paper](https://arxiv.org/abs/2408.05418) · [MATLAB integration](matlab/README.md) ·
[Model and paper correspondence](docs/PAPER.md) · [Measured results](docs/RESULTS.md)

ExoBalanceLab connects MATLAB control research to reproducible MuJoCo experiments on
ankle-exoskeleton standing balance. It extends the original
[standing_balance_sim](https://github.com/celwell20/standing_balance_sim) work with one
installable Python package, explicit controller contracts, physics checks and an
articulated human model.

The research question comes from **Raz et al., “Ankle Exoskeletons May Hinder Standing
Balance in Simple Models of Older and Younger Adults” (2024)**: assistance can help in
some states and reduce the ability to recover in others. This project makes both
outcomes visible and keeps the data behind them.

![Synchronized human and exoskeleton simulation](docs/media/human.gif)

[Watch the full human comparison](docs/media/human.mp4) ·
[Inspect assistance-induced contact-constraint violations](docs/media/paper_constraints.mp4)

## Three contributions

1. **A practical MATLAB → MuJoCo bridge.** Validated CSV/MAT v7/v7.3 trajectories,
   logarithmic-time interpolation, explicit sample-and-hold timing, separate physics/control
   rates, and a persistent MATLAB Engine adapter. The original one- and two-joint entry
   points now use the same replay implementation. HJB value-function exports provide
   torque-rate feedback on the lifted state.
2. **A paper-linked validation workbench.** Five Table I profiles, independently checked
   Eq. (3) dynamics, human MT/MRTD limits, GC/PD exoskeletons, foot constraints, full
   recovery-grid records and an unconstrained-foot MuJoCo comparison. The scope of each
   result is stated alongside it.
3. **An articulated human–device extension.** Bilateral hip, knee and ankle joints,
   distributed body mass, device mass/inertia, actual ground contact, chest perturbations,
   and synchronized videos with human/device torque and contact information.

**Current evidence:** numerical dynamics agreement, tested controller transport and
controller-specific recovery experiments. The published Hamilton–Jacobi stabilizable
regions and Table II are a separate reproduction milestone; this repository does not
label a PD recovery grid as a maximal reachable set. See [paper correspondence](docs/PAPER.md).

## Run in five minutes

```bash
git clone https://github.com/HaoweiLi1/ExoBalanceLab.git
cd ExoBalanceLab
conda env create -f environment.yml
conda activate exobalancelab

# Headless numerical comparison; no MATLAB license or display required.
exobalance compare --model paper --profile OF --out logs/first_run

# Full bilateral human, identical 80 N chest pulse for each assistance mode.
# Linux headless rendering: EGL requires a working graphics driver.
MUJOCO_GL=egl exobalance compare --model human --lean 0 --velocity 0 \
  --push 80 --duration 6 --video --out logs/human_demo
```

With a desktop display, omit `MUJOCO_GL=egl`. On machines without EGL, use
`MUJOCO_GL=osmesa` after installing the operating system's OSMesa library. Rendering is
optional: numerical runs do not create a GUI. Reusing an output directory is rejected
so a failed experiment cannot overwrite an earlier recording.

## Bring your MATLAB controller

Export desired **human** torques, with time in seconds and torque in Nm:

```matlab
addpath('matlab');
time_s = (0:0.001:6)';
torque_nm = 8*sin(2*pi*0.5*time_s).*exp(-time_s);
export_controller('controller.mat', time_s, torque_nm);
```

```bash
exobalance run --trajectory controller.mat --lean 0 --velocity 0 \
  --control-period .005 --out logs/replay
```

For feedback, install the Engine package from **your MATLAB release**, then run:

```bash
python -m pip install /path/to/MATLAB/extern/engines/python
exobalance run --matlab-function standing_pd --matlab-directory matlab \
  --control-period .01 --out logs/matlab_feedback
```

The simulation clock does not wait for wall-clock deadlines. MATLAB feedback latency is
measured; this interface does not promise hard real-time operation. Offline replay works
without MATLAB installed. The example `standing_pd.m` assumes no assistance; external
controllers must account for the configured exoskeleton themselves.

See [MATLAB integration](matlab/README.md) for channel ordering, timing, format validation,
HJB export, saturation, and runtime validation status.

## Choose the model you need

| Model | Degrees of freedom | Contact | Purpose |
|---|---:|---|---|
| `paper` | 1 | Fixed foot; analytically required ground reactions | Match Eq. (3) and evaluate C1–C5 |
| `contact` | 4 | Foot slides, lifts and rotates under MuJoCo contact | Test failure of the fixed-foot assumption |
| `human` | 9 | Independent feet; sagittal floating pelvis | Explore joint coordination and device loading |

All models use SI units, world **x forward, y left, z up**. Paper lean is
`q = theta - pi/2`: **positive q is forward toward the toes**; positive ankle torque is
dorsiflexion, negative is plantarflexion. Human joint channels and frame conventions are
specified in [the API guide](exobalancelab/README.md).

The human model is a torque-driven sagittal anthropometric approximation. It has no
muscle recruitment, neural delay, lateral balance or subject-specific identification.
The extra geometry is not evidence of physiological validation. The same physical device
mass remains attached in all human comparison modes; `none` means its motor is off.

## Reproduce the evidence

```bash
pytest -q
ruff check exobalancelab tests
python -m exobalancelab.validation --out logs/numerical_validation
python -m exobalancelab.sweep --points 13 --out logs/recovery_grid
```

Each run includes a standalone `model.xml`, `config.json`, `metrics.json`, full `episode.npz`,
a scalar `signals.csv`, and source/model hashes in `provenance.json`. Replay additionally
stores the actual input samples. Optional videos are streamed to FFmpeg without buffering
all full-resolution frames. [Published results](docs/RESULTS.md) explain the experiment
selection, pass criteria, observed failures and timing scope.

## Repository map

```text
exobalancelab/        Unified models, controllers, simulation, rendering and CLI
matlab/               Export helpers and persistent-feedback example
tests/                Transport, timing, mechanics, contact and model contract tests
docs/                 Paper correspondence, results and branch provenance
joint1_humanoid/      Historical ankle workflow and original MATLAB optimization scripts
joint2_humanoid/      Historical ankle–hip workflow and recordings
logs/                 Local generated runs (ignored by Git)
```

`main`, `backup` and `origin` have one integrated history. The latter two were already
ancestors of `main`; [migration notes](docs/MIGRATION.md) identify exactly what is retained
and how to recover older muscle/control examples. The maintained entry point is
`exobalance`; historical recordings keep their original provenance.

## Research context and attribution

- [Raz et al. (2024)](https://arxiv.org/abs/2408.05418) and the authors'
  [AnkleExoBRS](https://github.com/Raquelometer/AnkleExoBRS) provide the central question,
  lifted dynamics and reachability method.
- [MuJoCo](https://github.com/google-deepmind/mujoco) provides contact dynamics and rendering.
- [MyoSuite](https://github.com/MyoHub/myosuite) is a relevant musculoskeletal comparison
  and a future validation target. This project's articulated model is independently built;
  it does not include MyoSuite meshes or muscles.
- Original simulation work: [celwell20/standing_balance_sim](https://github.com/celwell20/standing_balance_sim).
  Haowei Li's ankle/exoskeleton and ankle–hip extensions are retained in Git history.

See [NOTICE](NOTICE.md) for provenance and licensing scope. Cite the paper for its
scientific method and [CITATION.cff](CITATION.cff) when referring to this software.
Contributions that add measured controller comparisons, physiological parameters, or
reproducible MATLAB/HJB evidence are welcome; see [contributing](CONTRIBUTING.md).
