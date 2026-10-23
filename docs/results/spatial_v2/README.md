# Spatial diagnosis evidence

This dataset follows [the spatial diagnosis](../../SPATIAL_CONVERGENCE.md). Physical goals
and acceptance thresholds match the original validation; historical inputs are unchanged.
The two preselected unassisted forward families are YF/source-entry and OM/common-ankle.
**The spatial gate is not accepted.** These pilots do not evaluate GC-effect convergence.

There are 31 three-state snapshots (27 new solves and four historical references),
19 declared comparisons, and six separate analytic-reference runs. Both 337³ cases
completed; their 225³→337³ area/IoU checks fail. Software checks comprise 89 basic,
eight real-MATLAB and eight JAX tests; see [verification](verification.json).

| Contents | Meaning |
|---|---|
| `manifest.json` | Complete case/comparison lists, frozen thresholds and input hashes |
| `solves/` | Actual solver summaries, physical targets/models and compact projections |
| `sources/` | Deduplicated solver snapshots named by SHA-256 |
| `preparation_sources/` | MATLAB preparation snapshots, also checked by hash |
| `geometry_inputs/` | Actual centerline samples used to measure target representation |
| `analytic_reference/` | Independent double-integrator fields, exact-set parameters and source snapshots |
| `diagnostics/` | Preparation/coordinate checks and predeclared axis, mapping and precision protocols |
| `report/` | Rebuilt areas, comparisons, geometry, analytic errors and figure |

Rebuild without MATLAB, CUDA, 3D histories or local experiment files:

```bash
python -m examples.report_spatial --out logs/rebuilt_spatial
```

The report verifies file hashes, complete analytic cohorts and identical physical
contracts within each pair. It also compares parsed solver syntax: the original OM
pair has different byte hashes because of formatting, with identical syntax trees.
`INCOMPLETE_GC` would mean area/IoU passed but the full spatial gate remains unevaluated;
`NOT_MET` records a failed area/IoU check. Time and precision comparisons use their own
explicit thresholds. No failed comparison is replaced by an accepted method.

Area has units rad²/s. Grids are recorded as both requested and actual sizes, including
the extra nodes required for equilibrium alignment. CPU/GPU wall times reflect shared
workstation load; these records are accuracy experiments, not speed benchmarks.
Full prepared fields and unsuccessful exploratory scripts remain under local `logs/`.
Rerun instructions, source/dependency pins and the exact frozen-goal example are in
[Spatial convergence](../../SPATIAL_CONVERGENCE.md) and [GPU HJB](../../HJB_GPU.md).
