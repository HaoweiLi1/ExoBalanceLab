# Remaining acceptance

- Reconcile the specific upstream source/parameter/target-set issues in `PAPER.md` and
  reproduce the original five-profile HJB figures/Table II with a convergence study.
  MATLAB runtime and the helperOC/ToolboxLS, YALMIP/SeDuMi and polyhedra dependencies now
  run. The nominal source and reconciled GC/PD paths now execute, with checked dynamics,
  regenerated assisted targets and short forward rollouts. Quantifying the effects of
  source/target reconciliation and full region convergence remains open. A certified
  invariant target is a separate stronger extension.
  With 101 angle nodes, a 0.5 rad/s buffer and 10 ms history, all six predicted-inside
  states in the original nominal cohort reach the unchanged targets. The sampled
  forward projected area still changes by 9.21% from 51 angle nodes. Complete spatial
  convergence, long-horizon assisted solves and broader forward validation.
- Resolve the inherited source licensing scope with relevant rights holders before
  claiming a project-wide open-source license.
- Validate any future musculoskeletal/3D or subject-specific extension against independent
  physiological evidence; the current articulated model is a sagittal engineering model.
