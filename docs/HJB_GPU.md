# HJB experiments on GPU

This optional backend integrates the fields prepared by the same MATLAB code used for
the native experiments. MATLAB supplies the dynamics, target boxes, contact constraints,
scales and human torque-rate bounds. JAX supplies the numerical integration. Backend
agreement and spatial convergence are measured separately.

## Environment

```bash
conda env create -f environment-hjb.yml
conda activate exobalancelab-hjb
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The separate environment uses Python 3.12, pinned JAX/CUDA 13 and pinned
[StanfordASL/hj_reachability](https://github.com/StanfordASL/hj_reachability/tree/9aebaf97440672e18b6e205cbf1d947a44f41079).
The standard MuJoCo environment remains on Python 3.11. CUDA 13 requires a compatible
NVIDIA GPU and Linux driver 580 or newer; see [JAX installation](https://docs.jax.dev/en/latest/installation.html).
CPU execution is an explicit `--platform cpu` choice, useful for small verification cases.

## Prepare, solve, validate

With the [MATLAB dependencies](../matlab/README.md#prepared-hjb-dependencies) installed:

```matlab
addpath('matlab');
solve_paper_hjb('/path/to/AnkleExoBRS', 'logs/hjb_dependencies', ...
    'logs/prepared_OF', resolution=[101;269;51], prepare_only=true);
```

Preparation writes `problem.mat`, the source/model snapshot and a summary marked
`prepared`. It performs no HJB integration. Assisted cases use `exo="gc"` or `exo="pd"`;
population and velocity direction are selected with `profile` and `direction`.
For matched assistance comparisons, `align_zero=true` samples zero velocity/torque in
all modes. It extends each uniform axis by at most one cell per side; the physical
torque/contact constraints remain enforced and actual grid dimensions are recorded.

```bash
python -m exobalancelab.hjb_solver \
  --problem logs/prepared_OF/problem.mat --out logs/gpu_OF

python -m exobalancelab.hjb_validation --run-dir logs/gpu_OF \
  --out logs/gpu_OF_validation --protocol docs/results/hjb_nominal_1p4/protocol.json
```

The default saves the complete value history for time-dependent feedback. Output is
HDF5 with the same SI fields as the MATLAB exporter, readable by `HJBPolicy`; use
MATLAB `h5read` to inspect Python-produced files. `--projection-only` saves the final
three-dimensional field and its torque projection, without a feedback history.
Each run records its prepared-input SHA-256, package versions, device, compiled temporary
memory and progress. Existing output directories are never overwritten.
The history is named `policy.partial.mat` during integration and becomes `policy.mat`
only after it is complete and closed. Interrupted runs retain their partial data.

## Numerical correspondence

- Float64, ENO3 with right-biased exact ties, global Lax–Friedrichs dissipation,
  SSP-RK3 and CFL factor 0.8.
- The CFL bound sums each axis's global maximum, matching ToolboxLS.
- Constraints are applied after each internal RK3 step; `none` and `minVWithV0`
  retain their separate reachable-set and reachable-tube meanings.
- The internal endpoint tolerance is helperOC's 1e-4 in nondimensional time.
- The prepared fields and contract are used directly. No assistance law is inferred
  from filenames or replaced during a solve.

To compare against a native MATLAB solve with identical inputs:

```bash
python -m exobalancelab.hjb_solver_comparison \
  --reference logs/native_OF --candidate logs/gpu_OF --out logs/backend_comparison
```

The comparison checks initial fields, axes, targets and physical contracts before
reporting every saved time's value error and sign disagreement, final projected areas,
policy hashes and an overlaid figure. Floating-point stencil choices can produce local
differences; agreement of projected areas is not bitwise equivalence or a convergence proof.
Recorded experiments are reported in [Results](RESULTS.md).
