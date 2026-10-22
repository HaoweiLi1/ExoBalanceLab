# Remaining acceptance

- Complete runtime sign-in for MATLAB R2026b: activation was reported complete, but the
  subsequent batch check requested sign-in and exited with `Sign-in failed`; the desktop
  also remains at its sign-in dialog. Matching Engine and Optimization Toolbox are installed.
- Run `EXOBALANCE_MATLAB_TEST=1 pytest -q -s tests/test_matlab_runtime.py`, record real
  MATLAB exports, closed-loop parity and Engine latency, then update the runtime claim.
- Reconcile the specific upstream source/parameter/target-set issues in `PAPER.md` and
  reproduce the original five-profile HJB figures/Table II with a convergence study.
  Four numerical dependency sources are downloaded and pinned in `matlab/README.md`;
  acquire the polyhedra utility and validate SeDuMi/MEX and the solver in MATLAB.
- Resolve the inherited source licensing scope with relevant rights holders before
  claiming a project-wide open-source license.
- Validate any future musculoskeletal/3D or subject-specific extension against independent
  physiological evidence; the current articulated model is a sagittal engineering model.
