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
