# Validation v1 evidence

The [validation guide](../../VALIDATION.md) defines the goals, predeclared tolerances,
measured outcomes and interpretation. [report/summary.json](report/summary.json)
is the complete numerical summary; all cases remain included, including failures.

| Inputs | Contents |
|---|---|
| `hjb/` | Frozen 70-case manifest, solved summaries and value projections; 68 prepared target files |
| `preparation_sources/`, `solver_sources/` | Deduplicated MATLAB and Python sources used to prepare/solve the fields |
| `preparation_index.json` | Case-to-source and target mapping |
| `paired/` | 480 primary paper/human outcomes, protocols and source hashes; six selected video traces |
| `timing/` | All 234 timing evaluations; original and held-command endpoint reassessment records |
| `yf_validation/` | All 36 YF inside/boundary/outside evaluations |
| `paired_refinement/`, `contact_followup/` | Twelve additional episodes diagnosing the unstable contact condition |
| `paired_design_coupled*` | Superseded exploratory data; controller design varied with physics step and is excluded from primary results |

`checksums.json` covers the 288 curated input files. Derived files under `report/`,
this index and the software/rebuild verification record are outside that input manifest.
Large value histories, prepared full fields, caches and unselected full traces remain local.
The six historical OF policies are needed to rerun the exact timing study; they are not
needed to rebuild its report. Original run paths and source hashes are historical records.
The two saved Python solver revisions differ only in line wrapping; their syntax trees
are identical. The [verification record](verification.json) retains the hashes and case counts.

From the repository root in the basic environment:

```bash
python examples/report_validation.py --input docs/results/validation_v1 --out logs/rebuilt_validation
```

This rebuild uses the included projections/outcomes and reruns the 20 small work-energy
ODE checks. It needs neither MATLAB nor the GPU HJB solver. Video reconstruction and
full experiment commands are in [Reproducing](../../REPRODUCING.md).
