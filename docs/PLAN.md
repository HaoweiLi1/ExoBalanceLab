# Delivery plan

ExoBalanceLab presents an existing research effort motivated by Daphna Raz's
standing-balance theory. The delivery goal is a public-ready research repository that
shows the modeling, control integration, experimental reasoning and simulation results.
A new paper or independent theoretical contribution is not required.

This plan defines the next work; unchecked acceptance items are not completed claims.
[TODO](TODO.md) tracks execution. The repository remains private until publication.

## Existing foundation

The maintained package already implements three MuJoCo models (`paper`, `contact`,
`human`), no-assistance/GC/PD comparisons, torque and torque-rate limits, MATLAB replay
and Engine feedback, HJB policy execution, and an optional JAX solver. Episode outputs
include configuration, model XML, trajectories, metrics and provenance. Rendering
produces synchronized comparison videos and plots.

[Results](RESULTS.md) records the completed dynamics checks, MATLAB comparisons,
population HJB studies, spatial/target sensitivity and articulated-model perturbations.
Full quantitative reproduction of the paper's stabilizable regions is not established;
the articulated model does not have a transferred HJB guarantee. Existing disagreements
and failed cases remain part of the evidence.

## Delivery goals and acceptance

| Workstream | Work to complete | Observable acceptance |
|---|---|---|
| Usable entry points | Verify clean Conda installation and the existing paper/human demos; provide a small documented CSV/MAT replay example | A fresh checkout runs the documented basic paths without MATLAB or local research folders; documented outputs are produced |
| Reproducible evidence | Connect selected claims and figures to committed inputs and exact commands; repair missing paths or small reproduction utilities where needed | Main summary figures/tables can be rebuilt from curated data; full simulations and HJB solves have separate dependency and input instructions |
| Research presentation | Keep README concise; identify the source theory, this repository's implementation work and measured limits; curate videos and figures | A visitor can understand the research question, inspect a paired simulation and follow a result to its evidence without reading the entire experiment log |
| Release readiness | Resolve inherited redistribution/licensing scope, inspect release assets and verify current checks | Release contents and attribution are reviewed, maintained checks pass, and any optional checks or remaining limitations are explicitly recorded |

### Functional completion

Use the existing `exobalance run` / `compare` interfaces and output format. Implementation
work is limited to defects or missing examples/utilities found while completing these paths:

1. **Basic simulation:** run paper-model and articulated-model comparisons with none/GC/PD
   assistance, save metrics and traces, and render the documented human demonstration.
2. **Controller exchange:** demonstrate a small CSV/MAT trajectory with channels, units,
   time coverage and command limits explained. Python replay must not require MATLAB.
3. **Evidence inspection:** reproduce the selected plots/tables from committed records;
   distinguish rebuilding a report from rerunning physics or solving a new HJB problem.
4. **Advanced integration:** provide a clear optional route to licensed MATLAB feedback
   and HJB generation/replay, including external inputs and their pinned sources. Keep
   MATLAB/JAX requirements out of the basic demo path.

During verification, check installed invocation outside the checkout as well as the
standard workflow: result provenance must identify the package source rather than an
unrelated current directory. Preserve existing interfaces, numerical results and
historical manifests when fixing release-path defects.

### Presentation outcome

The README should contain a short motivation, the main human-model comparison preview,
a compact capability summary, working quick-start commands, a few measured results,
documentation links and source/citation at the end. Retain a paper/contact comparison
as supporting media. Use existing experiments and consistent labels/units; keep detailed
campaign tables, failures and assumptions in the linked documentation.

A reader should be able to distinguish the paper's theory, inherited simulation code,
and this repository's implemented integration and experiments. Describe work supported
by commits and artifacts; the snapshot import is not evidence of sole authorship.

## Final repository structure

Keep the current package layout; add small examples or reproduction helpers only where
a documented path lacks them. This is the intended delivery structure, not a claim that
all examples and release checks already exist:

```text
README.md                  Concise research overview, demo and quick start
exobalancelab/             Maintained models, control interfaces, simulation and analysis
matlab/                   Optional MATLAB export, feedback and HJB integration
examples/                 Small runnable examples and reusable inputs, where needed
docs/
  PAPER.md                Theory correspondence, assumptions and source differences
  RESULTS.md              Measured outcomes and links to evidence/reproduction commands
  media/                  Selected previews, videos and figures
  results/                Curated configurations, metrics, traces and protocols
  PLAN.md                 Delivery goals and acceptance
  TODO.md                 Ordered work and optional future directions
  MIGRATION.md            Original commits and archive retrieval
  HJB_GPU.md              Optional GPU solver workflow
  README.md               Documentation navigation
tests/                    Mechanics, interfaces and optional backend checks
.github/workflows/        Automated maintained checks
environment*.yml          Basic and optional HJB Conda environments
pyproject.toml            Package, entry point and dependency definitions
NOTICE.md / CITATION.cff  Attribution, licensing scope and citation
third_party/              Required third-party notices
logs/                     Ignored local runs and large generated solver artifacts
```

The existing `joint1_humanoid/` and `joint2_humanoid/` directories remain compatibility
and historical material; tests still exercise their adapters. Review references before
any future relocation. `main` is the only development branch; the
`archive/exobalancelab-2026-10-23` tag preserves the full earlier history.

## Execution order and completion boundary

1. Verify the clean installation and basic examples; fix concrete failures.
2. Verify selected result-rebuild paths and curate the README/media links.
3. Review redistribution rights, release contents and current automated checks.
4. Present the release-ready snapshot and its validation record, then perform the
   separately authorized public visibility change.

Completion means the selected public workflows are reproducible and the presentation
accurately describes the evidence. It does not require rerunning every expensive
campaign, establishing full paper reproduction or implementing the long-term roadmap.
Stronger validation, additional controllers and muscle/3D models remain optional TODOs.
