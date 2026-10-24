# Transport convergence evidence

Compact evidence for the [transport follow-up](../../TRANSPORT_CONVERGENCE.md).
The [protocol](protocol.json) retains the original physical inputs and numerical
acceptance criteria. All computations ran on the development laptop's RTX 5060
Laptop GPU (8 GB); no remote compute was used.

| Record | Contents |
|---|---|
| `report/` | All native areas, assistance effects, 18 spatial comparisons and two time comparisons |
| `solved/` | 38 actual projections, physical inputs and per-run metadata |
| `inputs/` | The 18 frozen physical problems, reusable without MATLAB |
| `sources/`, `manifest.json` | Numerical source snapshots and input/projection hashes |
| `analytic_reference/` | Seven fields with exact-set, spatial and temporal checks |
| `physics/` | 18,000 sampled MATLAB-field comparisons and 68 enclosure-coverage checks |
| `coarse_reference/` | Failed 449³ analytic calibration, with independent report rebuild |
| `rejected_halo/` | Failed one-cell enclosure candidate, actual fields and sources |
| `allocator_comparison/` | Byte-identical analytic fields under platform/BFC allocation |
| `verification/` | Test logs, numerical source hashes and exact report/audit rebuild checks |

The common spatial pair is 561³→673³ at 10 ms. Time pilots compare 10→5 ms
at 673³. Both terminal goals are solved separately, then combined on the original
physical domain; identical goals are evaluated once. Targets and constraints do
not change with resolution. Projections use SI angle/velocity axes and minimize
over human torque. `report/areas.csv` includes signed GC area changes against the
matched unassisted solve, separately from numerical refinement errors.

Run metadata record actual steps, source and input hashes, device/library versions,
goal treatment and wall times. Per-run environment sidecars are included where
recorded. The first nine human solves used platform allocation; later solves used
BFC with the same numerical source. Interrupted attempts are not completed solves.
Timing observations are not real-time control benchmarks.

Use the [rebuild instructions](../../TRANSPORT_CONVERGENCE.md#reproduce) or the
[combined report helper](../../../examples/rebuild_results.py). Rebuilding reports
needs the basic Conda environment; solving again requires the separate HJB environment.
The compact records do not contain dense 3D histories or feedback policies. Earlier
MuJoCo trajectories and videos were not rerun with this backend. Acceptance concerns
the declared reduced-model projections, not exact Table II reproduction, invariant
terminal boxes or an articulated/hardware control guarantee.
