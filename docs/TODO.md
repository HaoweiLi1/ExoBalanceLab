# TODO

The goal is a clear, reproducible presentation of existing research motivated by
Daphna Raz's standing-balance theory. A new paper or independent theoretical contribution
is not a release requirement. The initial delivery work is verified; the four-gap
validation phase has been implemented and executed. Spatial convergence remains
unaccepted at the frozen tolerances. Model extensions remain optional.
Completed studies and their
limitations remain in [Results](RESULTS.md). Acceptance is defined in [the delivery plan](PLAN.md).

## Delivery work, in order

### 1. Reproducible entry points

- [x] Verify a fresh-checkout Conda installation and the README paper/human comparisons,
  including rendering requirements and expected output files.
- [x] Provide and check a small CSV/MAT replay example with documented units and channels;
  reuse the existing interface and keep Python replay independent of a MATLAB license.
- [x] Check installed invocation outside the checkout, including the recorded source
  identity, and fix concrete portability or provenance defects.

### 2. Evidence and presentation

- [x] Select the main summary figures/tables and verify their rebuild commands against
  committed inputs. Document separately any full reruns that need local-only HJB fields,
  external source checkouts, MATLAB or GPU dependencies.
- [x] Fill missing example/reproduction steps with small utilities where necessary;
  preserve existing measured results and historical manifests.
- [x] Review the main human video, supporting paper/contact comparison and result labels.
  Keep matched conditions, source records and failure outcomes inspectable.
- [x] Finalize the concise README and documentation navigation: motivation, implemented
  work, demo, quick start, measured results, limits, then source/citation.

### 3. Release readiness

- [x] Record the maintainer's confirmation of public distribution for inherited material,
  preserve attribution and component notices, and distinguish the implementation work
  from the paper's method. No project-wide license was added; see [NOTICE](../NOTICE.md).
- [x] Inspect tracked release contents, required notices, dependency instructions and
  links. Keep local environments, credentials and arbitrary generated runs excluded.
- [x] Run the maintained automated checks for the final code changes and record the
  demo/reproduction results. Report optional backend checks separately, including skips.
- [x] Publish the verified snapshot with the maintainer's authorization and confirm public access.

Verification details are in [Release verification](RELEASE.md).

## Four-gap validation

Protocol, numerical tolerances and reproduction commands are in [Validation](VALIDATION.md).
Execution completion and scientific acceptance are tracked separately.

- [x] Define explicit terminal goals, distinguish finite-time reach-avoid from invariance,
  and freeze spatial, temporal and rollout error thresholds before new outcomes.
- [x] Implement independent MT/MRTD through MATLAB, Python, policy contracts and CLI;
  check the independent limits against the real MATLAB runtime.
- [x] Complete the 70-case target/strength/grid/CFL campaign and curate every solve;
  internal-time checks pass 2/2, while spatial checks fail 18/18 at the frozen thresholds.
- [x] Implement causal command delay and physics-step torque limiting; execute all 234
  timing/refinement cases and retain misses, coverage failures and wall-clock measurements.
- [x] Implement paired fixed/readapted feedback and contact-preserving initialization;
  execute 480 episodes with fixed controller design across physics steps and render videos.
- [x] Diagnose the unstable human triplet at 62.5/31.25 µs with unchanged gains and
  contact criteria; all six follow-up episodes retain failure, and original records remain.
- [x] Implement and run independent work-energy checks; preserve the simplified model's
  assumptions instead of treating its envelope as a contact-constrained HJB region.
- [x] Finish all 36 additional YF inside/boundary/outside validations and publish the
  complete report with all failures; 18/18 predicted-inside states are accepted.
- [x] Isolate spatial error using frozen goals, independent box geometry, exact control
  flux, single-axis refinement and an analytic double-integrator reference; implement
  and run the candidate repairs in [Spatial convergence](SPATIAL_CONVERGENCE.md).

Scientific acceptance still requires evidence beyond execution:

- [ ] Establish spatial convergence at the frozen thresholds before claiming quantitative
  agreement with the article's region areas. Geometry/flux repairs and 337³ refinement
  retain 6.36% / 10.13% area differences for YF / OM. An accepted pilot must precede a new complete 18-family
  GC/MT/MRTD comparison; keep the same goals, ≤2% area, ≤1 pp GC effect and ≥0.98 IoU.
- [ ] Resolve the new Godunov candidate's OM time sensitivity before treating it as an
  accepted replacement: CFL 0.8→0.4 and 0.4→0.2 fail the unchanged IoU ≥0.995 check.
- [ ] Establish any stronger delay/coverage guarantee or hardware timing claim. The measured
  control-period and delay failures delimit the current implementation's tested behavior.

The paired articulated experiments do not establish harmful assistance or transfer the
reduced model's HJB guarantee. A positive result is not required or manufactured.
The exceptional contact condition fails at both finest tested steps; its precise
force-event timing remains sensitive to discretization.

## Optional long-term extensions

These directions are retained for future development, without a new-paper requirement.
They are not part of the current cleanup or prerequisites for public release.

- [ ] Compare additional controller families against the existing no-assistance, GC and
  PD baselines using matched tasks, human effort limits and recorded failure cases.
- [ ] Explore muscle-based actuation and subject-specific parameters, with independent
  physiological evidence for any claims beyond the current engineering model.
- [ ] Extend the articulated model to 3D balance and evaluate lateral perturbations,
  documenting the additional mechanics and validation assumptions.

See [scope](PLAN.md) and [paper correspondence](PAPER.md).
