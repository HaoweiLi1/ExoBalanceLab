# Validation

Run from the repository root in the project Conda environment:

```bash
pytest -q
ruff check exobalancelab tests
ruff format --check exobalancelab tests
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

`test_hjb_comparison.py` verifies complete outcome accounting and rejects comparisons
that drop failed cases or change initial states, targets or plant parameters.

`test_matlab_runtime.py` is opt-in: set `EXOBALANCE_MATLAB_TEST=1` after activation to
verify a real MATLAB CSV/MAT export and compare Engine feedback against the same Python
controller. Its default skip is visible in the test summary, not counted as a pass.

With `EXOBALANCE_HJB_SOURCE` and `EXOBALANCE_HJB_DEPS` set to the pinned dependency
checkouts, `test_matlab_assistance.py` compares 1,500 reconciled none/GC/PD states
against MuJoCo, checks SI/scaled equivalence, and verifies all 20 GC/PD target boxes.
It also checks that prepared GPU inputs exactly preserve a native solve's initial field.

In the separate [GPU environment](../docs/HJB_GPU.md), `EXOBALANCE_JAX_TEST=1 pytest -q
tests/test_hjb_solver.py` checks independent MATLAB ENO stencils (including tied
differences), exact linear advection, set/tube semantics and streamed policy interchange.
These checks are distinct from full numerical convergence and MuJoCo forward validation.

`test_hjb_regions.py` checks complete 30-case accounting, signed-velocity area integrals,
relative changes and rejection of different physical parameters using explicit synthetic
rectangular fields. The HJB reader tests also compare cached/uncached derivatives exactly
and reject a mismatched or damaged disk cache.
