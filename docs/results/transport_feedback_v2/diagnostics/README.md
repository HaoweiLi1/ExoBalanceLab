# Residual-failure diagnostics

Six fine-physics runs reused OM/none cases 01 and 05 and YF/none case 03. Each tried:

1. The primary bounded-rate interpolant at a 0.5 ms control period.
2. Extremal-rate selection from the interpolated unit switching function at the
   original 1 ms control period.

None achieved valid target reach. Five queries left coverage; OM 01 with extremal
selection completed but missed the 1 Nm torque half-width (final torque 1.382 Nm).
The original 1 ms campaign remains the reported experiment. These diagnostics do
not establish an exact cause or repair, and none is counted as an accepted result.

`results.json` retains every outcome, exact configuration, failed queries and policy
hashes. `diagnose_reader.py` is the actual executable source, archived with its hash
in `protocol.json`. The one complete trajectory is retained under `OM_1_switching/`.
