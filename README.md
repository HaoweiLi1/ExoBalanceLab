<div align="center">

# ExoBalanceLab

<p><strong>From standing-balance theory to human–exoskeleton simulation</strong></p>

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB)](environment.yml)
[![MuJoCo 3.3.7](https://img.shields.io/badge/MuJoCo-3.3.7-F28C28)](pyproject.toml)

[Source paper](https://arxiv.org/abs/2408.05418) · [Results](docs/RESULTS.md) ·
[MATLAB integration](matlab/README.md) · [Model correspondence](docs/PAPER.md)

</div>

![Human–exoskeleton standing balance in MuJoCo](docs/media/human.gif)

<p align="center">
The same 80 N chest perturbation, with assistance off, gravity compensation, and PD assistance.<br>
<a href="docs/media/human.mp4">Full simulation video</a> ·
<a href="docs/media/paper_constraints.mp4">Contact-constraint comparison</a>
</p>

## Overview

ExoBalanceLab began with the need to **test Daphna Raz's standing-balance theory in
simulation**. Its scientific foundation is Raz et al.,
[*Ankle Exoskeletons May Hinder Standing Balance in Simple Models of Older and Younger
Adults*](https://arxiv.org/abs/2408.05418). The central question is when ankle assistance
helps recovery—and when it reduces the ability to balance.

The project connects the paper's MATLAB reachability formulation to MuJoCo, then extends
the experiment to ground contact and an articulated human–exoskeleton model.

- **Theory to simulation:** lifted dynamics, five population profiles, human torque/rate
  limits, GC/PD assistance, and HJB torque-rate feedback.
- **MATLAB to MuJoCo:** CSV/MAT replay and persistent Engine feedback, with explicit units,
  control timing, and recorded raw/applied commands.
- **Inspect every experiment:** synchronized videos, state trajectories, contact margins,
  model snapshots, and reproducible comparisons.

## Quick start

```bash
git clone https://github.com/HaoweiLi1/ExoBalanceLab.git
cd ExoBalanceLab
conda env create -f environment.yml
conda activate exobalancelab

# Compare no assistance, gravity compensation, and PD on the paper model.
exobalance compare --model paper --profile OF --out logs/first_run

# Render the articulated human comparison shown above.
MUJOCO_GL=egl exobalance compare --model human --lean 0 --velocity 0 \
  --push 80 --duration 6 --video --out logs/human_demo
```

The quick-start comparisons run without MATLAB or a display. For desktop rendering, omit
`MUJOCO_GL=egl`; headless EGL needs a working graphics driver. Use a new output directory
for each run. [MATLAB setup and examples](matlab/README.md) cover replay, feedback and HJB solves.

## Models and validation

| Model | Experiment |
|---|---|
| `paper` · 1 DoF | Match the paper's dynamics and evaluate foot-contact constraints |
| `contact` · 4 DoF | Release the foot to study slipping, lifting and rotation |
| `human` · 9 DoF | Bilateral hips, knees and ankles, device inertia, and chest perturbations |

The paper model agrees with the analytical acceleration to **1.16e-9 rad/s²** across
1,500 states. Real MATLAB and Python feedback produce identical recorded positions and
commands in the six-second reference run. HJB forward validation and discretization
experiments are recorded in [Results](docs/RESULTS.md).

The completed spatial and target studies retain differences from the paper's stabilizable
regions; full reproduction is not established. The articulated model is a sagittal
engineering extension. [Paper correspondence](docs/PAPER.md) explains the assumptions and measured limits.

## Documentation

| Guide | Contents |
|---|---|
| [MATLAB integration](matlab/README.md) | Controller export, Engine setup, HJB generation and replay |
| [Models and API](exobalancelab/README.md) | Coordinates, units, model parameters and public interfaces |
| [Paper correspondence](docs/PAPER.md) | Equations, source audit and reproduction requirements |
| [Measured results](docs/RESULTS.md) | Experiments, failures, figures, videos and reproduction commands |
| [Validation](tests/README.md) · [History](docs/MIGRATION.md) | Tests and integration of the original branches |

## Source and citation

The theory and reachability method come from [Daphna Raz et al.](https://arxiv.org/abs/2408.05418)
and [AnkleExoBRS](https://github.com/Raquelometer/AnkleExoBRS). The simulation builds on
[standing_balance_sim](https://github.com/celwell20/standing_balance_sim), with its history
preserved. MuJoCo supplies the physics and rendering.

Cite the source paper for the scientific method and [CITATION.cff](CITATION.cff) for this
software. Attribution and inherited licensing scope are recorded in [NOTICE](NOTICE.md).
