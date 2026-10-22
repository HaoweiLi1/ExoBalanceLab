# Contributing

Use Python 3.11 and the Conda environment in `environment.yml`. Start with an issue or a
small pull request describing the observable behavior and how you measured it.

Before submitting a change:

```bash
pytest -q
ruff check exobalancelab tests
ruff format --check exobalancelab tests
```

Keep SI units, coordinate conventions and human/device torque separation explicit.
Test the affected behavior with meaningful mechanics or interface assertions. Preserve
failed experiments as well as successes. Do not commit environments, credentials,
MATLAB installations or arbitrary generated runs. Curated demonstration media and
result summaries belong in `docs/` with the command and parameter provenance that
produced them; full local runs belong in `logs/`.

For scientific contributions, distinguish a controller-specific simulation, a formal
reachability result and a physical experiment. Specify subject/model parameters, initial
states, action limits, sample rates and selection criteria before comparing outcomes.
An additional human geometry must have explicit mass/inertia and a stated modeling role.

Good next contributions include licensed MATLAB end-to-end validation, reconciled HJB
reproduction, subject-specific inertial parameters, muscle-based actuation and 3D balance.
Each needs evidence beyond adding an interface or increasing model complexity.
