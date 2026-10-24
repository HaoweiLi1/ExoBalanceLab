# Rejected overlap-domain reader

This candidate was stopped after **15 recorded coverage errors**. The remaining
planned outcomes are not inferred or included as completed experiments.

It evaluated one fixed physical point in both adjacent time grids. For example,
OM/none case 00 at 0.009 s remaining had human torque **2.234 Nm**. The earlier
slice had bounds ±2 Nm, but the current space-time cell spans approximately
**−4.088 to 8.129 Nm**. The old reader incorrectly rejected that valid cell.
[The recorded counterexample](../verification/moving_cell.json) verifies the
coordinates against the actual exported grids.

`protocol.json`, both cohort files, all 15 result records, and the actual source
snapshots are retained. The final campaign uses exactly those same cohort files
with reference-cell interpolation and unchanged goals and error limits.
