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
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

The separate environment uses Python 3.12, pinned JAX/CUDA 13 and pinned
[StanfordASL/hj_reachability](https://github.com/StanfordASL/hj_reachability/tree/9aebaf97440672e18b6e205cbf1d947a44f41079).
The standard MuJoCo environment remains on Python 3.11. CUDA 13 requires a compatible
NVIDIA GPU and Linux driver 580 or newer; see [JAX installation](https://docs.jax.dev/en/latest/installation.html).
CPU execution is an explicit `--platform cpu` choice, useful for small verification cases.

The recorded 301-grid run exhausted the default BFC allocator on an 8 GB GPU before its
first saved interval. The [JAX platform allocator](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
releases unused allocations. A complete 225-grid replay with this setting produced an
identical final field and file hash, with a sampled peak of 2,104 MiB. Its integration took
301.05 s versus 301.37 s for the original allocation setting on this workstation;
performance on other workloads may differ. The first complete 301-grid solve then took
878.35 s of integration with a sampled peak of 4,424 MiB. The remaining grid studies use
this explicit configuration. [Failure and equality records](results/hjb_grid_study/allocator_repair/) ·
[Successful 301-grid run](results/hjb_grid_study/OF_none_forward_301/).
The compiled-memory check is an estimate, not a guarantee against runtime allocation failure.

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
Independent capacities use `mt_scale` and `mrtd_scale` in `(0,1]`. Select an explicit
`target_definition` when reducing either capacity: `source_entry`, `recomputed` or
`common_ankle`. Their physical meanings and frozen validation thresholds are in
[the validation campaign](VALIDATION.md); the default `legacy` target remains nominal-only.

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
  SSP-RK3 and default CFL factor 0.8. `--cfl 0.4` refines internal solver time steps
  without changing the saved-history interval or prepared fields.
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
    if isfield(c, 'requested_resolution')
        resolution = c.requested_resolution;
    else
        resolution = protocol.requested_resolution;
    end
    if isfile(fullfile(c.prepared, 'problem.mat'))
        continue; % Reuse this protocol's previously prepared input.
    end
    solve_paper_hjb('/path/to/AnkleExoBRS', 'logs/hjb_dependencies', c.prepared, ...
        profile=c.profile, exo=c.exo, direction=c.direction, ...
        resolution=resolution, duration_s=protocol.horizon_s, ...
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
import hashlib
import json
import subprocess
import sys
from pathlib import Path

protocol = json.loads(Path('docs/results/hjb_paper225/protocol.json').read_text())
for case in protocol['cases']:
    problem = Path(case['prepared']) / 'problem.mat'
    output = Path(case['solved'])
    if output.exists():
        summary = json.loads((output / 'summary.json').read_text())
        assert summary['stage'] == 'solved', f'Incomplete run retained: {output}'
        with problem.open('rb') as file:
            assert summary['problem_sha256'] == hashlib.file_digest(file, 'sha256').hexdigest()
        assert not case['save_history'] or (output / 'policy.mat').exists()
        continue
    command = [sys.executable, '-m', 'exobalancelab.hjb_solver',
               '--problem', str(problem),
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
151³, 225³ and 301³, for every assistance/direction pair. The MATLAB loop above selects
each case's `requested_resolution`; use this protocol path to prepare the additional inputs.
The horizon, target construction, method and zero alignment
remain unchanged. Five 225³ runs are reused from the population campaign. The first
nominal forward run predates the atomic-output fix; it is repeated with identical inputs
under the common solver revision, with the original retained for an equality check.
[Source diff and revision record](results/hjb_grid_study/baseline_revision_audit.json).
Save only final fields for additional runs. Use the serial Python loop above with this
grid-study protocol; it reuses completed runs only when their prepared-input hashes match.

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
The completed OF study includes every projection and a report-only protocol, so its
[figures and all 12 comparisons can be rebuilt without a GPU](RESULTS.md#complete-of-spatial-study).

For a targeted study, explicitly declare `mode_direction_pairs` in the protocol; every
listed pair must appear at every declared resolution. Without this field the report
requires all six pairs. The [OM GC follow-up](results/hjb_om_gc_followup/protocol.json)
uses `[["none", "forward"], ["gc", "forward"]]` to investigate its 225-grid sign
disagreement with Table II. The same preparation, solve and `--grid-study` commands
apply with that protocol path. This is a follow-up selected after observing that result.
Its [completed six-case report](RESULTS.md#targeted-om-spatial-follow-up) also includes
projections and a report-only reconstruction command.

## Source-entry target diagnostic

The [OM target protocol](results/hjb_om_target_override/protocol.json) separately tests
the released assisted entry's fixed ankle bounds, ±0.01 rad/s and ±1 Nm. It was selected
after the OM/GC sign disagreement and source audit. Only these bounds and the initial
target field change; the computed ankle angle, toe target, physical model, constraints
and 225-grid solver settings stay fixed. This changes the terminal task, so its area
difference is a target sensitivity measurement.

After preparing the primary OM/GC/forward input, run in MATLAB:

```matlab
addpath('docs/results/hjb_om_target_override');
prepare_source_target_override('logs/hjb_paper225/prepared/OM_gc_forward', ...
    'logs/hjb_om_target_override/prepared/OM_gc_forward_225_source_bounds');
```

The [preparation check](results/hjb_om_target_override/preparation_check.json) confirms
exact equality of all ten other numerical arrays. The expanded ankle box passes the
[eight-corner check](results/hjb_om_target_override/corner_check.json); this does not
certify its controlled invariance. Use the primary 225-grid allocator settings for
this comparison, with one GPU worker at a time:

```bash
env -u LD_LIBRARY_PATH -u XLA_PYTHON_CLIENT_ALLOCATOR \
  -u XLA_PYTHON_CLIENT_MEM_FRACTION -u TF_GPU_ALLOCATOR -u XLA_FLAGS \
  XLA_PYTHON_CLIENT_PREALLOCATE=false python -m exobalancelab.hjb_solver \
  --problem logs/hjb_om_target_override/prepared/OM_gc_forward_225_source_bounds/problem.mat \
  --out logs/hjb_om_target_override/solved/OM_gc_forward_225_source_bounds --projection-only
```

With both primary OM reference runs complete, generate the comparison in the main
Conda environment:

```bash
python docs/results/hjb_om_target_override/report_target_override.py \
  docs/results/hjb_om_target_override/protocol.json logs/om_target_report
```

The [completed diagnostic](RESULTS.md#source-entry-target-diagnostic) includes the candidate
projection and a protocol pointing to all three retained projections. Rebuild its report
without MATLAB or a GPU:

```bash
python docs/results/hjb_om_target_override/report_target_override.py \
  docs/results/hjb_om_target_override/curated_protocol.json logs/om_target_report_rebuilt
```

The complete population report can also be regenerated from its included projections,
without MATLAB or a GPU; see the [results and reconstruction command](RESULTS.md#complete-225-grid-population-comparison).
