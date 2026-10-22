# Remaining acceptance

- Reconcile the specific upstream source/parameter/target-set issues in `PAPER.md` and
  reproduce the original five-profile HJB figures/Table II with a convergence study.
  MATLAB runtime and the helperOC/ToolboxLS, YALMIP/SeDuMi and polyhedra dependencies now
  run. The nominal baseline entry point uses reconstructed contact constraints; GC/PD
  source consistency, reproduction of published target approximations and full region
  convergence remain open. A certified invariant target is a separate stronger extension.
  The executed 1.4 s coarse OF/no-exo policy disagrees with preselected MuJoCo outcomes;
  refine spatial/time discretization and check the velocity domain before accepting it.
- Resolve the inherited source licensing scope with relevant rights holders before
  claiming a project-wide open-source license.
- Validate any future musculoskeletal/3D or subject-specific extension against independent
  physiological evidence; the current articulated model is a sagittal engineering model.
