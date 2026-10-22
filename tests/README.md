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

`test_matlab_runtime.py` is opt-in: set `EXOBALANCE_MATLAB_TEST=1` after activation to
verify a real MATLAB CSV/MAT export and compare Engine feedback against the same Python
controller. Its default skip is visible in the test summary, not counted as a pass.
