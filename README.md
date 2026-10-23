<div align="center">

# ExoBalanceLab

<p><strong>From standing-balance theory to human–exoskeleton simulation</strong></p>

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](environment.yml)
[![MuJoCo 3.3.7](https://img.shields.io/badge/MuJoCo-3.3.7-F28C28)](pyproject.toml)
[![Tests](https://github.com/HaoweiLi1/ExoBalanceLab/actions/workflows/tests.yml/badge.svg)](https://github.com/HaoweiLi1/ExoBalanceLab/actions/workflows/tests.yml)

<h3>
🎥 <a href="docs/media/human.mp4">Simulation</a> ·
📊 <a href="docs/RESULTS.md">Results</a> ·
🔗 <a href="matlab/README.md">MATLAB</a> ·
📚 <a href="docs/PAPER.md">Theory</a>
</h3>

</div>

![Human–exoskeleton standing balance in MuJoCo](docs/media/human.gif)

<p align="center">
The same 80 N chest perturbation, with assistance off, gravity compensation, and PD assistance.<br>
<a href="docs/media/human.mp4">Full simulation video</a> ·
<a href="docs/media/paper_constraints.mp4">Contact-constraint comparison</a>
</p>

## 🦿 Overview

ExoBalanceLab began with the need to **test Daphna Raz's standing-balance theory in
simulation**: when does ankle assistance help recovery, and when does it reduce the
ability to balance?

The project connects the paper's MATLAB reachability formulation to MuJoCo, then extends
the experiment to ground contact and an articulated human–exoskeleton model. This repository
collects the implementation, experiments and visualizations from that research effort.

- **Theory to simulation:** lifted dynamics, five population profiles, human torque/rate
  limits with independent MT/MRTD scaling, GC/PD assistance, and HJB torque-rate feedback.
- **MATLAB to MuJoCo:** CSV/MAT replay and persistent Engine feedback, with explicit units,
  sampled control, causal command delay, and recorded raw/applied commands.
- **Inspect every experiment:** synchronized videos, state trajectories, contact margins,
  model snapshots, and reproducible comparisons.

## 🚀 Quick start

```bash
git clone --depth 1 https://github.com/HaoweiLi1/ExoBalanceLab.git
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
Start with [CSV/MAT replay](examples/README.md) or [rebuild the recorded results](docs/REPRODUCING.md)
using the basic environment.

## 📊 Models and validation

| Model | Experiment |
|---|---|
| `paper` · 1 DoF | Match the paper's dynamics and evaluate foot-contact constraints |
| `contact` · 4 DoF | Release the foot to study slipping, lifting and rotation |
| `human` · 9 DoF | Bilateral hips, knees and ankles, device inertia, and chest perturbations |

| Recorded check | Result |
|---|---|
| Analytical dynamics · 1,500 states | Maximum acceleration error **1.16e-9 rad/s²** |
| MATLAB/Python feedback · six seconds | Identical recorded positions and commands |
| [HJB validation](docs/VALIDATION.md) · 70 solves | Time checks **2/2**; spatial convergence **0/18** at frozen tolerances |
| [Paired model study](docs/VALIDATION.md) · 480 episodes | 159/160 physics-refinement classifications agree; human harm not observed |

The completed spatial and target studies retain differences from the paper's stabilizable
regions; full reproduction is not established. The articulated model is a sagittal
engineering extension. [Results](docs/RESULTS.md) and [paper correspondence](docs/PAPER.md)
explain the assumptions, disagreements and measured limits.

## 📚 Documentation

| Guide | Contents |
|---|---|
| [MATLAB integration](matlab/README.md) | Controller export, Engine setup, HJB generation and replay |
| [Models and API](exobalancelab/README.md) | Coordinates, units, model parameters and public interfaces |
| [Paper correspondence](docs/PAPER.md) | Equations, source audit and reproduction requirements |
| [Measured results](docs/RESULTS.md) | Experiments, failures, figures, videos and reproduction commands |
| [Reproduce the results](docs/REPRODUCING.md) | Basic demos, offline report rebuilds and optional solver requirements |
| [Validation campaign](docs/VALIDATION.md) | Target/error standards, independent strength, timing and paired human models |
| [Validation](tests/README.md) · [History](docs/MIGRATION.md) | Tests and integration of the original branches |
| [Scope and TODO](docs/TODO.md) | Completed work, open numerical acceptance and optional extensions |

## 📄 Source and citation

The theory and reachability method come from [Daphna Raz et al.](https://arxiv.org/abs/2408.05418)
and [AnkleExoBRS](https://github.com/Raquelometer/AnkleExoBRS). The simulation builds on
[standing_balance_sim](https://github.com/celwell20/standing_balance_sim), whose original
commits and authorship are preserved in the [history archive](docs/MIGRATION.md).
MuJoCo supplies the physics and rendering.

Cite the source paper for the scientific method and [CITATION.cff](CITATION.cff) for this
software. Attribution and inherited licensing scope are recorded in [NOTICE](NOTICE.md).

```bibtex
@article{raz2024ankle,
  title  = {Ankle Exoskeletons May Hinder Standing Balance in Simple Models of Older and Younger Adults},
  author = {Raz, Daphna and Joshi, Varun and Umberger, Brian R. and Ozay, Necmiye},
  year   = {2024},
  eprint = {2408.05418},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2408.05418}
}
```
