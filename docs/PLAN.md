# Project scope

ExoBalanceLab presents an existing research effort motivated by Daphna Raz's
standing-balance theory. Its purpose is to make the implementation, experimental
reasoning and results easy to inspect and reproduce. Preparing this repository for
public display does not require a new paper or an independent theoretical contribution.

## Retained deliverables

- One maintained package and `main` branch, with Conda setup and a concise video-led README.
- The `paper`, `contact` and articulated `human` MuJoCo models, including GC/PD assistance.
- MATLAB replay, Engine feedback and HJB integration, plus the optional JAX solver.
- Existing numerical studies, successful and failed rollouts, comparison media and tests.
- Source attribution, paper correspondence and recoverable original Git history.

## Evidence and limits

[Results](RESULTS.md) records the completed dynamics checks, MATLAB comparisons,
population HJB studies, spatial/target sensitivity and human-model perturbation sweep.
[Paper correspondence](PAPER.md) explains what matches the source and what differs.
The evidence supports model-level observations; full quantitative reproduction of the
paper's stabilizable regions and a formal guarantee for the articulated model are not
established.

## Current boundary

Current work is repository cleanup and presentation of those results. Further experiments,
control changes and stronger validation are deferred to [TODO](TODO.md). New controller
families and muscle/3D models remain optional long-term directions, with no requirement
to produce a new paper. Public release is a separate remaining task; the repository
stays private during this cleanup.

Historical implementation milestones remain in `docs/PLAN.md` at the
`archive/exobalancelab-2026-10-23` tag; see [history retrieval](MIGRATION.md).
