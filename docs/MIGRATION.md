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
git ls-tree -r --name-only 2f446a1
git archive 2f446a1 muscle_humanoid | tar -x -C /path/to/historical-workspace
```

These snapshots are historical source, not the maintained launch interface. The old root
README incorrectly implied several removed directories still existed; the new README
points to their actual historical location.

The published independent repository is [HaoweiLi1/ExoBalanceLab](https://github.com/HaoweiLi1/ExoBalanceLab),
with only `main`. GitHub API verification confirmed `fork=false` and remote commit
`d0ed547783ae81e4a681309ceed376f1bbc7dabc` for the implementation release.
Pushing the preserved history to a normally created repository yields an independent
GitHub repository without a parent/fork network. The original fork is preserved as an
archive of prior work; deleting it is unnecessary for the new repository to be independent.
Original authorship remains in the commit graph and in [NOTICE](../NOTICE.md).
