# Attribution and licensing scope

ExoBalanceLab builds on the standing-balance simulation history from
<https://github.com/celwell20/standing_balance_sim> and the former
`HaoweiLi1/standing_balance_sim` fork, retired after migration. Original authorship is preserved in the
`archive/exobalancelab-2026-10-23` Git tag; see [history provenance](docs/MIGRATION.md).
The maintained `main` starts from an imported project snapshot. Its import commit records
the migration, not sole authorship of the inherited code, assets or research.
The inherited project described its origins in the Ozay Lab at the University of Michigan;
this independent extension does not imply institutional endorsement.

The project originated from the need to validate Daphna Raz's standing-balance theory
in simulation. Its scientific source is Raz, Joshi, Umberger and Ozay (2024), arXiv:2408.05418.
The paper's authors maintain <https://github.com/Raquelometer/AnkleExoBRS>.
ExoBalanceLab independently implements the documented mathematical model and preserves
explicit differences between the manuscript, released source and this simulator.

No root license was present in the inherited repository at inspection. This migration
has not unilaterally relicensed that source, historical assets or recordings. Public
availability and GitHub fork status do not settle redistribution/licensing permissions.
A project-wide open-source license remains a maintainer decision requiring the relevant
rights holders' permission for inherited components.

MuJoCo, NumPy, SciPy and other installed dependencies retain their respective licenses.
MATLAB and its toolboxes require the user's applicable MathWorks license and are not
redistributed. No MyoSuite models, third-party meshes, paper PDF or the authors' HJB source
are bundled into this repository.

The optional JAX solver uses StanfordASL's `hj_reachability`. Its ENO stencil routine
adapts that project's MIT-licensed implementation; the copyright notice and license are
preserved in [third_party/hj_reachability_LICENSE](third_party/hj_reachability_LICENSE).
MATLAB-prepared numerical fields keep the original physics separate from this solver.
