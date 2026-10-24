# Rejected one-cell initial enclosure

This candidate placed each physical terminal box one grid cell inside its initial
computational enclosure. It improved the independent reference but retained human-model
space/time errors. It was rejected in favor of the twice-half-width enclosure and
finer grids. These are historical failed comparisons, not accepted substitutes.

The five compact fields, physical inputs and actual source snapshots are retained.
`comparisons.json` uses the original area/IoU thresholds; GC-effect convergence was
not evaluated for this diagnostic subset. Rebuild from the repository root:

```bash
PYTHONPATH=. python docs/results/transport_v3/rejected_halo/rebuild.py
```
