# Reproduce the examples and results

Use the [basic Conda environment](../README.md#-quick-start) and run from the repository
root unless stated otherwise. Each output path must be new. These three workflows
have different inputs and purposes:

| Workflow | Inputs | Requirements | What it establishes |
|---|---|---|---|
| Run a demo | Packaged models and explicit initial conditions | Basic environment; graphics driver for video | Current implementation produces a fresh simulated episode |
| Rebuild a report | Committed projections, outcomes and episode arrays | Basic environment; no MATLAB/GPU solver | Recorded data reproduce the selected tables and figures |
| Solve HJB again | Frozen JSON inputs for transport, or MATLAB-prepared fields for the original backend | Separate JAX environment; MATLAB for new terminal SDP preparations | A new numerical solve, requiring its own validation |

## Run the basic demonstrations

```bash
exobalance compare --model paper --profile OF --out logs/first_run
MUJOCO_GL=egl exobalance compare --model human --lean 0 --velocity 0 \
  --push 80 --duration 6 --video --out logs/human_demo
```

The human demonstration uses the existing 80 N chest-push scenario. It was selected
after exploratory trials; the full [33-run force sweep](results/human_force_sweep/summary.json)
provides the surrounding outcomes. The supporting reduced-model comparison is:

```bash
MUJOCO_GL=egl exobalance compare --model paper --lean -.08 --velocity .2 \
  --duration 6 --video --out logs/paper_constraints
```

For desktop rendering, omit `MUJOCO_GL=egl`. Numerical runs and report plots need no
display; headless video rendering needs working EGL graphics support. Use the
[CSV/MAT replay example](../examples/README.md) to exercise offline controller exchange.

Each comparison creates `none/`, `gc/` and `pd/` directories with `config.json`,
`model.xml`, `metrics.json`, `episode.npz`, `signals.csv` and `provenance.json`.
`comparison.png` shows recorded lean and ankle torques. With `--video`, the root also
contains `comparison.mp4`, `comparison.gif` and `comparison_poster.png`.
For HJB/Engine runs, additional controller records are described in the API guide.

## Rebuild the selected evidence without solving again

```bash
python examples/rebuild_results.py --out logs/rebuilt_results
# Optionally render the existing recorded states, without rerunning their dynamics:
MUJOCO_GL=egl python examples/rebuild_results.py --video --out logs/rebuilt_with_video
```

The helper calls the existing report functions with the included curated protocols:

| Output | Recorded inputs | Result |
|---|---|---|
| `population/` | 30 population projections | Absolute areas, 20 Table II comparisons, PNG/PDF figures |
| `of_grid/` | 18 OF grid-study projections | Areas, 12 grid comparisons, PNG/PDF figures |
| `om_grid/` | Six OM follow-up projections | Areas, four grid comparisons, PNG/PDF figures |
| `target_diagnostic/` | OM reference and changed-target projections | Three areas and target-sensitivity metrics |
| `time_comparison/` | Three frozen-cohort outcome sets | All 27 outcomes, summary and PNG/PDF matrix |
| `transport/` | Frozen transport projections and analytic references | All 18 spatial comparisons, two time pilots, exact-reference checks and figures |
| `human_signals.png`, `paper_constraints_signals.png` | Six complete episode records | Original lean and human/device torque curves |

`--video` adds MP4/GIF/poster files for those same two comparisons. Graphics/library
versions may change video encoding or typography; compare numerical records rather
than expecting byte-identical media. Report paths and PDF metadata may also differ.
The transport reports are rebuilt in a copy of the curated dataset and checked against
the recorded numerical summaries. Rebuilding reports does not perform new integration
or eliminate the recorded disagreement with the paper. Individual commands are in
[Results](RESULTS.md).

## Rerun advanced integrations

The [transport follow-up](TRANSPORT_CONVERGENCE.md#reproduce) includes independent
analytic references, the frozen human-model inputs and a compact MATLAB-sample audit.
Its report/figure commands use the basic environment. Replaying the physical audit
needs JAX on the CPU; repeating the numerical integrations uses the separate HJB
environment. Neither operation needs a MATLAB license or the large preparation files.

The [spatial diagnosis](SPATIAL_CONVERGENCE.md) has a separate compact dataset. Rebuild
its geometry, area/IoU comparisons, analytic reference and figure in the basic environment:

```bash
python -m examples.report_spatial --out logs/rebuilt_spatial
```

The [four-gap validation campaign](VALIDATION.md) adds explicit target definitions,
independent MT/MRTD, control-period/delay sweeps and paired articulated experiments.
Its compact recorded inputs can be rebuilt without MATLAB or a GPU:

```bash
python examples/report_validation.py --input docs/results/validation_v1 \
  --out logs/rebuilt_validation
MUJOCO_GL=egl python examples/render_validation.py \
  --cases docs/results/validation_v1/paired/cases \
  --pair OF_human_fixed_dt0.0005_state07 --out logs/rebuilt_validation/human.mp4
```

The report rebuild also reruns the small independent work-energy ODE checks. Full HJB
and MuJoCo reruns use the separate commands in the validation guide. Large histories
and gradient caches are intentionally excluded from the curated dataset.

[MATLAB integration](../matlab/README.md) specifies Engine installation, process-local
Linux runtime configuration, author-source/dependency versions, policy generation and
forward validation. [GPU HJB](HJB_GPU.md) describes the separate Python 3.12/JAX
environment and MATLAB-prepared inputs. The basic environment does not install either
MATLAB Engine or JAX.

The full four-dimensional HJB value histories, prepared `problem.mat` files, downloaded
dependencies and derivative caches are generated under ignored `logs/`. They are not
included in a fresh clone. Follow the solve/preparation commands before running commands
that reference these files. Curated report protocols instead use the committed
`docs/results/` projections and summaries and need none of those large inputs.

Historical manifests keep their original machine paths and source hashes. They are
records of prior execution. Current examples use relative paths, and newly saved
episodes identify the installed source checkout even when called from another directory.
For a wheel installation or a system without Git, Git metadata is null; source/model
hashes remain available. [Tests](../tests/README.md) separates ordinary checks from
optional licensed MATLAB and GPU checks.
