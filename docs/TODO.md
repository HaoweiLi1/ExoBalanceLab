# TODO

The goal is a clear, reproducible presentation of existing research motivated by
Daphna Raz's standing-balance theory. A new paper or independent theoretical contribution
is not a release requirement. Items below are recorded for later work, not an active
experiment queue. Completed studies and their limitations remain in [Results](RESULTS.md).

## Before public release

- [ ] Confirm redistribution rights and licensing scope for inherited code, recordings
  and assets; retain source attribution and distinguish the simulator's contributions
  from the paper's method. See [NOTICE](../NOTICE.md).
- [ ] Check a fresh-clone Conda setup, the README demo and documented result-reproduction
  paths. Identify any large local-only inputs needed to regenerate curated results.
- [ ] Review the selected videos, figures and result links for a compact public showcase;
  retain numerical disagreements and failures alongside successes.
- [ ] Review the final release contents and explicitly switch repository visibility to
  public when ready. The documentation cleanup itself leaves visibility private.

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
