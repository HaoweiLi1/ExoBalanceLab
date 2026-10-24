# Coarse analytic calibration

The 449³, 10 ms bounded-jerk reference fails the unchanged 2% area / 0.98 IoU
requirements. It is retained as a failed calibration, outside the accepted
561³/673³ cohort. This repeat uses the exact frozen numerical source of the
accepted cohort; its manifest records the actual field and source hashes.

Rebuild the measured errors from the repository root in the basic environment:

```bash
PYTHONPATH=. python docs/results/transport_v3/coarse_reference/rebuild.py
```

The archived generator records how the field was obtained. The source snapshots
are historical execution evidence; use the maintained benchmark for new accepted
cohorts.
