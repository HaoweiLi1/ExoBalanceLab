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

For many forward rollouts, add `--gradient-cache logs/OF_gradient_cache` to
`hjb_validation`. The first invocation writes an uncompressed float64 derivative;
later queries read only the required pages through a memory map. The cache is bound
to the policy's SHA-256, and its own checksum is checked before reuse. It uses the same
two-slice interpolation as the default reader. Allow disk space equal to the uncompressed
value history (about 13 GB for a 225³ grid with 141 saved times).

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

## Complete population comparison

The [recorded protocol](results/hjb_paper225/protocol.json) fixes all 30 combinations,
solver settings, saved-history choices and the Table II reference values. From the
repository root, prepare its inputs in MATLAB with the source/dependency paths adjusted
for your machine. Use empty output paths; existing experiment files are preserved.

```matlab
addpath('matlab');
protocol = jsondecode(fileread('docs/results/hjb_paper225/protocol.json'));
for k = 1:numel(protocol.cases)
    c = protocol.cases(k);
    solve_paper_hjb('/path/to/AnkleExoBRS', 'logs/hjb_dependencies', c.prepared, ...
        profile=c.profile, exo=c.exo, direction=c.direction, ...
        resolution=protocol.requested_resolution, duration_s=protocol.horizon_s, ...
        output_step_s=protocol.output_step_s, ...
        velocity_buffer_rad_s=protocol.velocity_buffer_rad_s, ...
        method=protocol.method, align_zero=protocol.align_zero, prepare_only=true);
end
```

In the GPU environment, run one case at a time. OF saves all six policy histories for
forward validation; the other profiles save final fields and projections. Allow about
100 GB of free disk for prepared inputs, outputs and a few optional derivative caches.

```bash
python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

protocol = json.loads(Path('docs/results/hjb_paper225/protocol.json').read_text())
for case in protocol['cases']:
    command = [sys.executable, '-m', 'exobalancelab.hjb_solver',
               '--problem', str(Path(case['prepared']) / 'problem.mat'),
               '--out', case['solved']]
    if not case['save_history']:
        command.append('--projection-only')
    subprocess.run(command, check=True)
PY
```

After every run is complete, aggregate the same protocol:

```bash
python -m exobalancelab.hjb_regions \
  --protocol docs/results/hjb_paper225/protocol.json --out logs/paper_region_report
```

The report checks the five profiles, three assistance modes, both velocity directions,
physical parameters and common solver settings. It refuses an incomplete campaign.
Outputs include absolute areas, relative changes against each profile's no-assistance
baseline, differences from Table II, hashes, and PNG/PDF/SVG figures. Grid convergence
and MuJoCo forward validation remain separate experiments.

## Spatial-resolution study

The [OF protocol](results/hjb_grid_study/protocol.json) fixes three requested grids,
151³, 225³ and 301³, for every assistance/direction pair. Prepare each case with its
own `requested_resolution`; the horizon, target construction, method and zero alignment
remain unchanged. The six 225³ runs are reused from the population campaign. Save only
final fields for the additional cases.

Once all 18 recorded cases are solved:

```bash
python -m exobalancelab.hjb_regions --grid-study \
  --protocol docs/results/hjb_grid_study/protocol.json --out logs/of_grid_report
```

This checks the declared resolutions, all six pairs, identical targets and solver code.
It reports each native-grid area, then interpolates coarse projected values onto finer
nodes inside their common domain. Common-grid sign agreement and area estimates separate
value-field changes from quadrature/domain differences. No extrapolation is used.
CSV/JSON and PNG/PDF/SVG outputs retain actual axis extents and projection hashes.
The report measures spatial sensitivity; it does not automatically certify convergence.
