# TODO

The goal is a clear, reproducible presentation of existing research motivated by
Daphna Raz's standing-balance theory. A new paper or independent theoretical contribution
is not a release requirement. The delivery work below is the next implementation scope;
scientific validation and model extensions remain deferred. Completed studies and their
limitations remain in [Results](RESULTS.md). Acceptance is defined in [the delivery plan](PLAN.md).

## Delivery work, in order

### 1. Reproducible entry points

- [ ] Verify a fresh-checkout Conda installation and the README paper/human comparisons,
  including rendering requirements and expected output files.
- [ ] Provide and check a small CSV/MAT replay example with documented units and channels;
  reuse the existing interface and keep Python replay independent of a MATLAB license.
- [ ] Check installed invocation outside the checkout, including the recorded source
  identity, and fix concrete portability or provenance defects.

### 2. Evidence and presentation

- [ ] Select the main summary figures/tables and verify their rebuild commands against
  committed inputs. Document separately any full reruns that need local-only HJB fields,
  external source checkouts, MATLAB or GPU dependencies.
- [ ] Fill missing example/reproduction steps with small utilities where necessary;
  preserve existing measured results and historical manifests.
- [ ] Review the main human video, supporting paper/contact comparison and result labels.
  Keep matched conditions, source records and failure outcomes inspectable.
- [ ] Finalize the concise README and documentation navigation: motivation, implemented
  work, demo, quick start, measured results, limits, then source/citation.

### 3. Release readiness

- [ ] Confirm redistribution rights and licensing scope for inherited code, recordings
  and assets; retain attribution and describe the repository's implementation work
  separately from the paper's method. See [NOTICE](../NOTICE.md).
- [ ] Inspect tracked release contents, required notices, dependency instructions and
  links. Keep local environments, credentials and arbitrary generated runs excluded.
- [ ] Run the maintained automated checks for the final code changes and record the
  demo/reproduction results. Report optional backend checks separately, including skips.
- [ ] Present the release-ready snapshot and remaining limitations; switch to public
  only after the separate publication decision. Keep visibility private during preparation.

## Optional follow-up validation

These items would support stronger scientific claims. They are not prerequisites for
presenting the existing work with its measured limits.

- [ ] Reconcile source/manuscript terminal-target definitions and report matched-target
  comparisons. The OM gravity-compensation sign depends on this choice; changing the
  target changes the task. Keep numerical reproduction separate from any claim that
  a terminal box is control invariant.
- [ ] Extend spatial/time refinement and quantify region-area uncertainty. The completed
  population campaign and target diagnostic do not establish convergence or full
  quantitative agreement with Table II.
- [ ] Separate maximum human torque (MT) from maximum torque rate (MRTD) in parameter
  studies, and compare the paper's independent sweeps and work–energy analysis.
- [ ] Broaden forward validation beyond the existing OF cohort, including boundary
  states and policy coverage failures, with all selected outcomes retained.
- [ ] Evaluate realistic control periods and delay sensitivity; fine-step numerical
  successes alone do not establish real-time feasibility.
- [ ] If testing transfer to the articulated model, predefine paired initial states,
  perturbations and fixed versus readapted human feedback. Its current assistance sweep
  does not establish the paper's harmful-assistance effect in that model.

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
