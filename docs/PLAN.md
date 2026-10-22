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
  remains pending successful activation.

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
