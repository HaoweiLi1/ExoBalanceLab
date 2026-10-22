# Remaining acceptance

- Complete MathWorks account activation for the discovered MATLAB R2026b installation.
  Matching Python Engine and Optimization Toolbox are installed; batch startup currently
  reports MathWorks licensing error -1.2. The activation GUI is open for the user.
- Run `EXOBALANCE_MATLAB_TEST=1 pytest -q -s tests/test_matlab_runtime.py`, record real
  MATLAB exports, closed-loop parity and Engine latency, then update the runtime claim.
- Reconcile the specific upstream source/parameter/target-set issues in `PAPER.md` and
  reproduce the original five-profile HJB figures/Table II with a convergence study.
- Resolve the inherited source licensing scope with relevant rights holders before
  claiming a project-wide open-source license.
- Validate any future musculoskeletal/3D or subject-specific extension against independent
  physiological evidence; the current articulated model is a sagittal engineering model.
