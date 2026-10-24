# Native transport feedback evidence

All experiments ran on the current computer. Four native 673³ / 10 ms policy exports
exactly match the accepted transport projections. The complete repaired-reader campaign
contains **72 rollouts**, with **30 completed**, **28 valid target reaches**, and **42
coverage errors**. The strict refinement gate passes **15/36 pairs**; the remaining
pairs are incomplete. Three predicted-inside cases still fail. See the
[measured report](../../TRANSPORT_FEEDBACK.md) for interpretation and unchanged limits.

| Contents | Meaning |
|---|---|
| `protocol.json`, `selection_*.json` | Frozen conditions and exact shared initial states |
| `summary.json`, `outcomes.csv` | Every outcome and the original acceptance gate |
| `episodes/` | All 72 result records; 30 complete traces, XML, metrics and provenance |
| `policies/` | Four physical inputs, final projections, export hashes and actual source snapshots |
| `sources/`, `sources.json` | Actual repaired-reader rollout sources and their hashes |
| `environment.json` | Local Python, libraries, CPU, GPU and driver |
| `rejected_overlap_reader/` | First reader, stopped after 15 recorded errors; no unrun outcomes inferred |
| `diagnostics/` | Six unsuccessful targeted checks, plus their executable source |
| `figures/` | Matched video selection, plots and rendering source |
| `verification/` | Tests, exact cohort reuse and report reconstruction |
| `checksums.json` | SHA-256 of every other curated file |

Complete trajectories are stored at 1 ms intervals. Metrics use the full 12.5/6.25 µs
observations and the actual post-step endpoint. The four policy tables remain local:
2.917, 2.511, 4.038 and 5.007 GB for OM none, OM GC, YF none and YF GC respectively.
The first export used CPU decision extraction; the other three used equivalent slabbed
device extraction with full initial/final checks. Each retains its actual sources.
Rollout-reader sources are recorded separately from export sources.

To rebuild the report without policies, MATLAB, new physics or a GPU:

```bash
cp -a docs/results/transport_feedback_v2 logs/feedback_report_copy
python -m examples.validate_transport_feedback --report-only \
  --out logs/feedback_report_copy
```

To render the recorded states:

```bash
MUJOCO_GL=egl python -m examples.render_transport_feedback \
  --source logs/feedback_report_copy --out logs/feedback_video_copy
```

[Full policy generation and experiment commands](../../TRANSPORT_FEEDBACK.md#reproduce)
use new output directories. The dense tables are unnecessary for inspecting outcomes
or regenerating videos; regenerating them requires the optional local solver environment.
