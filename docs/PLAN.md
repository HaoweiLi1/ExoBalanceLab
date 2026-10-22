# ExoBalanceLab implementation and acceptance

Requested outcome: a standalone repository integrating `main`, `backup` and `origin`,
with a Conda environment, MATLAB controller interfaces, paper-linked MuJoCo validation,
an articulated human/exoskeleton extension, videos and clear reproducible documentation.

## Source inspection, October 22, 2026

- Starting tree: clean, `main` at `48c9f04`. Remote fetched successfully.
- `backup` at `23e63c1` and `origin` at `2f446a1` are already ancestors of `main`.
  Historical models removed from the current tree remain in Git history. Preserve them and
  document retrieval; do not resurrect stale implementations as competing entry points.
- Parent: `celwell20/standing_balance_sim`; GitHub confirms `fork=true`.
- Paper: <https://arxiv.org/abs/2408.05418>, v3, October 2, 2024.
- Authors' source: `Raquelometer/AnkleExoBRS`, commit
  `517ab41b7767c5ab0e856a8885b118bd739a5011`.
- No root license was found. Preserve attribution and history; do not apply a new license
  to inherited work without rights-holder permission.
- MATLAB was absent from PATH but later located at `~/Applications/MATLAB/R2026b`.
  The matching Python Engine and Optimization Toolbox are installed. Batch startup returned
  MathWorks licensing error -1.2; activation UI was opened for the user. Runtime validation
  was pending at that stage; it subsequently passed after program-level sign-in.

## Work and evidence required

1. Establish one packaged entry point, isolated Conda installation and deterministic logging.
2. Implement validated CSV/MAT replay, MATLAB export, explicit units/timing and a persistent
   optional MATLAB Engine feedback adapter. Test errors, interpolation, timing, action limits.
3. Implement Table I profiles, Eq. (3), GC/PD assistance, C1-C5 diagnostics, compare MuJoCo
   acceleration and integration to an independent ODE. Keep controller-specific recovery
   experiments distinct from the paper's Hamilton-Jacobi stabilizable regions.
4. Implement articulated bilateral human dynamics with device inertia, contact, perturbations,
   human/exoskeleton torque logging and paired video. Validate mass, actuator and contact contracts.
5. Publish measured results, provenance, API/usage instructions, comparison media and tests.
6. Deliver a new standalone `HaoweiLi1/ExoBalanceLab` repository with one `main` branch,
   preserving history. Verify its remote SHA and `fork=false`. The old repository need not be
   deleted to establish the new independent repository.

## Remaining scientific acceptance

The request for a complete paper validation includes published HJB stabilizable regions;
dynamics agreement and sampled controller rollouts alone do not prove that result. The
authors' MATLAB/helperOC dependencies and missing referenced source functions must be
resolved or reported, with an explicit path to running and comparing the full experiment.
The articulated extension is a model study, not physical or clinical validation.

## Delivered implementation milestone

- Conda Python 3.11 environment created; native MuJoCo and optional matching MATLAB Engine installed.
- Unified interfaces/models, historical bridge fixes, tests, paired MP4/GIF media and evidence committed.
- Corrected forward-angle/pressure-center conventions checked independently; final results regenerated.
- 33 tests pass, 1 licensed MATLAB test is explicitly skipped. Lint, formatting, local Markdown links,
  dependency consistency and an installed CLI invocation from another directory pass.
- GitHub repository created with preserved history, one main branch and `fork=false`; implementation
  commit `d0ed547783ae81e4a681309ceed376f1bbc7dabc` pushed and remote SHA matched.
- [Implementation CI run](https://github.com/HaoweiLi1/ExoBalanceLab/actions/runs/39784619849) passed.
- Full requested scientific acceptance remains open as detailed in `TODO.md`; this milestone does
  not claim complete HJB reproduction.

## MATLAB execution milestone

- MATLAB R2026b batch startup and Optimization Toolbox license verified after sign-in.
- Real CSV/MAT v7 and v7.3 exports, persistent feedback parity and contact-constraint
  checks pass. A six-second comparison records all 600 Engine call latencies and paired
  MATLAB/Python episodes; the measured deadline misses are reported in `RESULTS.md`.
- YALMIP/SeDuMi and the polyhedra utility pass real numerical checks. The nominal HJB
  runner uses unchanged author dynamics/targets with reconstructed contact constraints.
- A 0.1 s native-parameter nominal solve and nine preselected MuJoCo rollouts exercise
  the complete HJB/export/feedback chain. All outcomes, including a policy coverage
  failure, are retained. The source box targets are not controlled invariant.
- Full five-profile no-exo/GC/PD regions, target/controller reconciliation, convergence
  and broader MuJoCo forward validation remain required.
- A 1.4 s OF nominal solve completed in 1,003.732 s. Nine preselected matching MuJoCo
  rollouts yielded one valid target reach, four terminal misses and four grid-coverage
  errors. This coarse-grid prediction fails forward acceptance; all outcomes are retained.
- Fixed SI time-endpoint roundoff and reran the identical selected states. The full
  regression suite passes 54 tests, including all three real MATLAB/source tests.

## Publication preparation and temporal refinement

- Repository visibility is private at the user's request. Subsequent commits retain
  the maintainer identity; inherited authorship is preserved without adding contributors.
- README follows the maintainer's CrashFactory/End2Race presentation: one main video,
  concise setup and documentation links, with Daphna Raz's theory stated as the origin.
- The original nine HJB states remain frozen. A 0.5 rad/s velocity buffer with
  51×269×51 nodes and 10 ms history yields five valid target reaches, one angular
  target miss and three coverage errors. Decimating the same value history to
  100 ms yields zero valid reaches; shared slices are bit-for-bit equal.
- The paired comparison exports all outcomes to CSV/JSON and PNG/PDF. It rejects
  changed cohorts, plants or targets and omitted failures. All 62 tests pass,
  including three real MATLAB/source checks; spatial convergence remains open.

## Assisted HJB implementation, October 23, 2026

- Shared nominal/GC/PD solve/export path added. Assisted dynamics evaluate physical
  controller gains and ±50 Nm saturation before nondimensionalization; the original
  nominal entry point retains its source defaults and stored targets.
- Recomputed 20 assisted target boxes with the Appendix II SDP, equilibrium-relative
  constraint distances, linear box containment and nonlinear corner-contact checks.
- Real MATLAB/MuJoCo comparison passes 1,500 states across five profiles/three modes
  with maximum derivative error 1.40e-9. Full suite: 65 passed, four native MATLAB checks.
- GC and PD short solves and all preselected outcomes are retained. A PD ankle-target
  miss is traced to extremal-rate sampling: 1 ms and 100 μs miss, while 50/25 μs reach
  the unchanged target. All three inside-group PD states pass in both finer cohorts.
- Added explicit physics/control periods to frozen-cohort validation and a synchronized
  video of the earlier 100/10 ms value-history comparison. Full regional reproduction
  and spatial convergence are still required.
- The nominal 101×269×51 solve finishes in 2,552.523 s. All six predicted-inside
  frozen states now reach the original targets with valid contact. The three outside
  failures are unchanged; the sampled projected area changes by 9.21%, so a larger
  201-angle-node run is started for the next convergence check.
