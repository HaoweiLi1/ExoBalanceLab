# Branch integration and provenance

The starting repository was `HaoweiLi1/standing_balance_sim`, a GitHub fork of
`celwell20/standing_balance_sim`. Remote metadata and all branches were refreshed on
October 22, 2026 before implementation.

| Original branch | Tip at inspection | Integration |
|---|---|---|
| `main` | `48c9f04` | Starting point of the maintained package |
| `backup` | `23e63c1` | Already an ancestor of `main`; ankle–hip functionality retained |
| `origin` | `2f446a1` | Already an ancestor of `main`; original simulator provenance retained |

There were no divergent commits requiring a conflict merge. A merge commit with no new
content would not improve integration. The package combines maintained model, transport,
logging and rendering paths while preserving the original one-/two-joint directories and
all tracked historical recordings. Both historical precomputed controllers now delegate
trajectory loading/interpolation to the same tested bridge.

Earlier models removed by the old refactor are still recoverable from their original
commit, including `initial_humanoid`, `final_humanoid`, `muscle_humanoid`, `test_model_motor`,
control examples, images and videos. For example:

```bash
git fetch origin
git ls-tree -r --name-only origin/legacy/origin
git archive origin/legacy/origin muscle_humanoid | tar -x -C /path/to/historical-workspace
```

These snapshots are historical source, not the maintained launch interface. The old root
README incorrectly implied several removed directories still existed; the new README
points to their actual historical location.

## Branch organization — October 23, 2026

The maintained repository is [HaoweiLi1/ExoBalanceLab](https://github.com/HaoweiLi1/ExoBalanceLab),
an independent, private repository. All three branches from
[HaoweiLi1/standing_balance_sim](https://github.com/HaoweiLi1/standing_balance_sim)
are retained here under `legacy/`, with their original commit IDs and authors:

| Current branch | Source | Preserved commit |
|---|---|---|
| `main` | Maintained project snapshot, followed by new development | Independent history from October 23, 2026 |
| `legacy/main` | Original repository's `main` | `48c9f04f6fa73d098f069f285e24beb0f1c0adf8` |
| `legacy/backup` | Original repository's `backup` | `23e63c1a88d54ab07802a55335c38004ceb44f77` |
| `legacy/origin` | Original repository's `origin` | `2f446a1e374fb829e52013870c6c2871040cb265` |
| `archive/exobalancelab-2026-10-23` | ExoBalanceLab before the history separation | `a8717646ab8a53c3e73685814114b52e985838ad` |

The original repository and its branches remain available. The archive contains the full
implementation and validation history, including commit IDs referenced by experiment
manifests. All original branches are ancestors of that archive tip; no source work was
lost or overwritten during integration.

At the maintainer's request, the default `main` begins with a parentless import of the
current project. Only this README and provenance documentation changed during the import;
code, environments, tests, numerical results and simulation media are identical to the
archive. The import author records the migration and does not replace the original
contributors' authorship, which remains in the archived commits and [NOTICE](../NOTICE.md).

GitHub calculates the contributors graph from default-branch history. This separation
keeps historical contributors in the archive while starting the maintained branch's
contribution history afresh. GitHub says contributor displays can take about 24 hours to
refresh after history changes; see [GitHub's contributor documentation](https://docs.github.com/en/repositories/viewing-activity-and-data-for-your-repository/viewing-a-projects-contributors).

Use `main` for current development. Inspect historical models through `legacy/*`; merging
those branches back into `main` would also reintroduce their historical authors into its
commit graph. Existing clones should preserve local changes before switching to the new
history; a fresh clone is the simplest option. The archive remains available for comparing
or recovering any prior work.
