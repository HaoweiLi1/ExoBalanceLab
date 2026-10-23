# Validation

Run from the repository root in the project Conda environment:

```bash
pytest -q
ruff check exobalancelab tests examples
ruff format --check exobalancelab tests examples
```

`test_bridge.py` covers MATLAB-compatible CSV/MAT v7/v7.3 data, temporal sampling,
coverage failures, multirate control, legacy adapters and synthetic HJB torque-rate feedback.
`test_physics.py` compares native MuJoCo accelerations to an independent equation across
five profiles, verifies integration convergence, static CoP/signs, actuator mass/rate
contracts and real supporting contact in the articulated model.

These tests use native MuJoCo, with no renderer or MATLAB license. Synthetic MAT/HJB tests
verify interchange semantics and are explicitly not MATLAB execution or original-paper
reachability evidence. Numerical campaigns and demonstration videos use separate commands
listed in the root README. They are not rerun as part of every unit test.

`test_provenance.py` checks that results identify the installed source checkout when
called from an unrelated repository. Wheel installations and systems without Git
retain source/model hashes with null Git metadata.

`test_hjb_comparison.py` verifies complete outcome accounting and rejects comparisons
that drop failed cases or change initial states, targets or plant parameters.

`test_matlab_runtime.py` is opt-in: set `EXOBALANCE_MATLAB_TEST=1` after activation to
verify a real MATLAB CSV/MAT export and compare Engine feedback against the same Python
controller. Its default skip is visible in the test summary, not counted as a pass.

With `EXOBALANCE_HJB_SOURCE` and `EXOBALANCE_HJB_DEPS` set to the pinned dependency
checkouts, `test_matlab_assistance.py` compares 1,500 reconciled none/GC/PD states
against MuJoCo, checks SI/scaled equivalence, and verifies all 20 GC/PD target boxes.
It also checks that prepared GPU inputs exactly preserve a native solve's initial field.
The independent-capacity check executes MATLAB at nominal, half-MT and half-MRTD
settings, checking effective bounds, scaled solver inputs and both source-entry
velocity target widths.

In the separate [GPU environment](../docs/HJB_GPU.md), `EXOBALANCE_JAX_TEST=1 pytest -q
tests/test_hjb_solver.py` checks independent MATLAB ENO stencils (including tied
differences), exact linear advection, set/tube semantics and streamed policy interchange.
These checks are distinct from full numerical convergence and MuJoCo forward validation.

The spatial checks add exact concave Hamiltonian extrema, thin-interval transport,
mapped physical advection, and frozen-goal MATLAB/Python parity. `test_spatial_diagnostics.py`
checks the analytic double-integrator bounds against independently optimized acceleration
sequences and rejects relaxed acceptance thresholds. The numerical reference campaign runs
separately with `python -m examples.hjb_analytic_reference`; its measured errors remain
visible in [Spatial convergence](../docs/SPATIAL_CONVERGENCE.md).

`test_hjb_regions.py` checks complete 30-case accounting, signed-velocity area integrals,
relative changes and rejection of different physical parameters using explicit synthetic
rectangular fields. The HJB reader tests also compare cached/uncached derivatives exactly
and reject a mismatched or damaged disk cache.
The grid-study test samples one exact linear field at three resolutions: its native
indicator quadrature changes while common-grid boundaries agree exactly. It also rejects
changed targets, changed solver code and an omitted declared resolution. Targeted studies
must explicitly declare their pairs; omitted, extra or duplicated pairs are rejected.

The HJB validation sampler excludes physically invalid torque nodes introduced by grid
alignment. Frozen-protocol tests retain invalid historical states and their recorded
rejections rather than silently replacing them.

`test_validation_controls.py` checks independent capacity limits in actual MuJoCo runs,
causal command arrival, physics-step rate limits at slower control periods, and matching
initial contact/gains across paired human conditions and physics refinements. It also
checks that terminal contact uses the delivered device torque when commands are delayed.
`test_validation_reporting.py` rejects incomplete cohorts, including missing failed runs,
and compares target and contact classifications separately during physics refinement.
`test_work_energy.py` checks the SI energy implementation against independent quadrature.
The full [validation campaign](../docs/VALIDATION.md) runs separately and retains numerical
disagreements even when these software tests pass.
