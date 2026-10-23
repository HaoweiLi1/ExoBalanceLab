# Controller replay

Run these commands from the repository root in the basic Conda environment. They
generate equivalent CSV and MATLAB-compatible MAT files using Python; MATLAB is not
required. Use a new output directory each time.

```bash
python examples/make_replay.py --out logs/replay_input
exobalance run --model paper --exo none --lean 0 --velocity 0 --duration 1 \
  --trajectory logs/replay_input/torque.csv --out logs/replay_csv
exobalance run --model paper --exo none --lean 0 --velocity 0 --duration 1 \
  --trajectory logs/replay_input/torque.mat --out logs/replay_mat
```

The input contains 101 samples from 0 to 1 second, including both endpoints, with one
human ankle torque channel in Nm. It is a 0.5 Nm sinusoid for checking interchange,
not an optimized recovery policy. Linear interpolation samples the current simulation
time; add `--interpolation zoh` to hold each sample instead. Commands still pass through
the human torque and torque-rate limits. Exoskeleton torque is applied separately.

Each run writes `none/config.json`, `model.xml`, `episode.npz`, `signals.csv`,
`metrics.json` and `provenance.json`, plus `comparison.png` in the output root.
CSV and MAT replay produce identical state/command arrays for this example. Compare
the traces, not execution-time metrics or metadata containing input paths.

`paper` and `contact` accept one channel. The `human` model requires six channels:
left hip/knee/ankle, then right hip/knee/ankle. The MAT fields are `time_s` (N-by-1)
and `torque_nm` (N-by-C); CSV contains time followed by C torque columns.
See [MATLAB integration](../matlab/README.md) for licensed Engine feedback and HJB use.

## Validation experiments

The [validation guide](../docs/VALIDATION.md) defines targets, tolerances and exact commands.

| Script | Purpose |
|---|---|
| `validation_hjb.py` | Freeze, prepare and solve the target/MT/MRTD/grid/CFL campaign |
| `report_spatial.py` | Rebuild the compact spatial diagnosis, geometry, analytic reference and figure |
| `hjb_analytic_reference.py` | Run LF/Godunov against an analytic double-integrator reachable set; JAX environment |
| `diagnose_spatial.py` | Measure initial goal interpolation and area quadrature from prepared fields |
| `spatial_convergence.py` | Prepare/solve the full fixed-target matrix; pilot results remain unaccepted |
| `validation_rollouts.py` | Run complete paired-model and HJB control-period/delay cohorts |
| `refine_human_contact.py` | Refine every human triplet with an unstable recovery classification |
| `report_validation.py` | Rebuild area, strength, timing, pairing and energy figures from recorded inputs |
| `curate_validation.py` | Copy compact complete evidence; exclude large policies and caches |
| `render_validation.py` | Render an explicitly selected recorded triplet with source hashes |

For a single independent-capacity/delay check in the basic environment:

```bash
exobalance run --model paper --mt-scale 0.7 --mrtd-scale 1 \
  --control-period 0.005 --command-delay 0.01 --out logs/independent_mt
```
