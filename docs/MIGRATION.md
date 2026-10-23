# History and provenance

ExoBalanceLab integrates the former `HaoweiLi1/standing_balance_sim` fork of
`celwell20/standing_balance_sim`. The original branches were already related by
ancestry; no divergent implementation needed a conflict merge.

| Original branch | Preserved tip | Role |
|---|---|---|
| `main` | `48c9f04f6fa73d098f069f285e24beb0f1c0adf8` | Starting point of the maintained package |
| `backup` | `23e63c1a88d54ab07802a55335c38004ceb44f77` | Ankle–hip implementation history |
| `origin` | `2f446a1e374fb829e52013870c6c2871040cb265` | Original simulator history |

## One branch, one history archive

The maintained repository is [HaoweiLi1/ExoBalanceLab](https://github.com/HaoweiLi1/ExoBalanceLab).
As of October 23, 2026, it retains one development branch and one archive tag:

| Reference | Purpose |
|---|---|
| Branch `main` | Maintained research package, results and documentation |
| Tag `archive/exobalancelab-2026-10-23` | Full implementation history at `a8717646ab8a53c3e73685814114b52e985838ad` |

All three original tips above are ancestors of the archive tag. The redundant `legacy/*`
branches and the former archive branch were removed after preserving this tag. Their
commits, authorship and experiment-manifest references remain recoverable.

The old fork was deleted on October 23, 2026 after its branch tips were verified against
migrated copies. At the maintainer's request, the new `main` starts from a parentless
project import. Code, environments, tests, numerical results and media matched the
archive at import; subsequent maintenance is recorded on `main`. The import records the
migration and does not replace original authorship; see [NOTICE](../NOTICE.md).
The repository became public on October 23, 2026 after the maintainer confirmed
publication and the [release checks](RELEASE.md) passed.

## Retrieve historical work

Earlier models such as `initial_humanoid`, `final_humanoid` and `muscle_humanoid`, along
with their control examples and recordings, remain in historical commits. For example:

```bash
git fetch origin tag archive/exobalancelab-2026-10-23
git ls-tree -r --name-only 2f446a1e374fb829e52013870c6c2871040cb265
mkdir -p /path/to/historical-workspace
git archive 2f446a1e374fb829e52013870c6c2871040cb265 muscle_humanoid | tar -x -C /path/to/historical-workspace
```

Use `main` and its packaged entry points for current experiments. Historical models are
reference material. The archive also preserves the earlier implementation milestones in
`docs/PLAN.md`; the current [scope](PLAN.md) describes the research showcase.

## Local project location

The local checkout is `/home/kaidi-2/Documents/ExoBalanceLab`, with `origin` set to
`https://github.com/HaoweiLi1/ExoBalanceLab.git`. Both project Conda environments were
relocated with Conda to retain their installed versions and update executable paths.
Editable package installations point to the renamed checkout. The former
`standing_balance_sim` directory is no longer the project path.

Historical experiment manifests retain their original execution paths as provenance.
They are records of prior runs, not current environment or checkout configuration.
