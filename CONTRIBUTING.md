# Contribution workflow

`main` must remain reproducible and pass all required checks. Development takes
place on a focused branch and reaches `main` through a pull request.

## Change discipline

- Keep each commit limited to one reviewable engineering decision.
- Do not commit source datasets, pixel-bearing evidence, local environments, or
  machine-specific IDE state.
- Keep experiment reports, compact evidence, and generators consistent in the
  same change.
- Preserve read-only access to external dataset trees.
- Update checksum manifests when an excluded generated artifact changes.

## Required local checks

Run the dataset-independent checks before opening a pull request:

```powershell
& .\.conda-env\python.exe -m pip check
& .\.conda-env\python.exe -m pytest -q
& .\.conda-env\Scripts\ruff.exe check .
& .\.conda-env\python.exe -m compileall -q src scripts tests
```

Run dataset-dependent scripts only when the change affects their behavior or
their evidence chain. Record those targeted checks in the pull request.

## Pull requests

Describe the problem, the decision, the boundary of the change, and the exact
checks performed. Merge only after every required check for the current commit
has completed successfully.
