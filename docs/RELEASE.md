# Release verification — October 23, 2026

This release completes the research showcase and reproduction workflows in [the delivery
plan](PLAN.md). It adds small replay/report examples and corrects experiment provenance
when the installed package is called from another directory. Existing physics, controllers
and numerical campaign records are retained.

## Verified workflows

A new Conda environment was created from `environment.yml` in a fresh GitHub clone.
The final code at `4323d07829e271d52d0bbcd2040d22626c48b292` was then installed and checked
from a second clean clone using those same declared dependencies. MATLAB Engine and JAX
were absent. Validation used Linux, Python 3.11.17 and MuJoCo 3.3.7.

| Check | Result |
|---|---|
| Basic paper comparison and 80 N human demo | All expected records produced; human MP4/GIF/poster rendered |
| Supporting paper/contact demonstrations | Executed successfully; recovery/constraint outcomes retained |
| Human and paper demonstration reruns | All 60 arrays per comparison exactly match the committed records |
| CSV versus MAT replay | All 20 state/command arrays match exactly |
| Reports rebuilt from committed data | 30 population areas, 18 OF areas, six OM areas and three target-diagnostic areas match |
| Numerical comparisons | All 20 Table II entries, 16 spatial comparisons and 27 temporal outcomes match |
| Installed invocation outside the source directory | Source commit recorded correctly; eight final-clone episodes record a clean checkout |
| Recorded-state video reconstruction | Human and paper comparisons render at 30 fps, six seconds, 1440×640 |
| Automated tests | 70 passed; eight optional MATLAB/JAX checks skipped |
| Lint and formatting | Passed for maintained code, tests and examples |

Report comparison excludes only expected local run paths and the hash of the substituted
curated protocol. The current temporal reporter also emits `rollout: null` for older
protocols. Numerical records are unchanged. Fourteen Matplotlib/pyparsing deprecation
warnings remain visible. Licensed MATLAB/JAX campaigns were not repeated for this release;
their earlier execution evidence remains in [Results](RESULTS.md).

[Machine-readable evidence and source hashes](results/release_20261023/summary.json) ·
[Implementation CI](https://github.com/HaoweiLi1/ExoBalanceLab/actions/runs/39918497073) ·
[Reproduction commands](REPRODUCING.md).

## Publication and scope

The repository was made public on October 23, 2026 with the maintainer's confirmation.
Original attribution,
the archive tag and component license notices are retained; no project-wide license was
added. See [NOTICE](../NOTICE.md). Tracked files exclude local environments and run folders.
README claims link to measured evidence, including disagreements and failures.

New controllers, muscle/3D models and stronger scientific validation remain optional
[TODOs](TODO.md). Full quantitative reproduction of the source paper, spatial convergence
and physiological validation of the articulated model remain unestablished.
