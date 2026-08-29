# fingerprint-new-method

Reproducible research workspace for developing and qualifying fingerprint
Level-3 primitives. The current work evaluates pore visibility, annotation
fitness, and measurement feasibility; it does not claim a production-ready pore
detector or fingerprint recognition system.

The repository is private research software. External datasets remain read-only
and are not part of the repository.

## Experiments

- [Experiment 001 protocol](docs/experiments/001-sd300-level3-feasibility-preregistered-selection.md)
- [Experiment 001 results](docs/experiments/001-sd300-level3-feasibility-results.md)
- [Experiment 002 results](docs/experiments/002-l3sf-pore-annotation-feasibility-results.md)
- [Experiment 003 protocol](docs/experiments/003-l3sf-annotated-final-crosswalk-protocol.md)

## External datasets

Set `FINGERPRINT_DATASETS_ROOT` to the directory that contains the external
dataset trees. If it is unset, scripts use a sibling directory named
`fingerprint-datasets` next to the repository checkout.

Expected layout:

```text
fingerprint-datasets/
├── NIST/
└── L3_SF_V2/
    └── L3SF_V2/
```

PowerShell override example:

```powershell
$env:FINGERPRINT_DATASETS_ROOT = 'D:\research-data\fingerprints'
```

Never commit source images. Review [the data and licensing policy](docs/data-and-licensing.md)
before publishing any derived artifact.

## Python environment

The logical environment name is `fingerprint-new-method`; its physical prefix
is the project-local `.conda-env/` directory. Python 3.12 packages come only
from `conda-forge`.

- `environment.yml` records the human-maintained dependency intent.
- `conda-lock.yaml` records the exact Windows package solution.
- `pyproject.toml` provides package metadata and PyPI-facing requirements.

Create or verify the environment from the lockfile:

```powershell
.\scripts\bootstrap_environment.ps1
```

Run commands without relying on shell activation:

```powershell
& .\.conda-env\python.exe -m pytest
& .\.conda-env\python.exe .\scripts\experiment_002_finalize_review.py
```

To intentionally resolve changed requirements, recreate from `environment.yml`
and export a new Windows lock from a Conda-enabled shell:

```powershell
.\scripts\bootstrap_environment.ps1 -Recreate -FromIntent
conda export `
  --prefix .\.conda-env `
  --file .\conda-lock.yaml `
  --override-channels `
  --channel conda-forge `
  --platform win-64
```

Do not install research dependencies into Conda `base`. The editable project
install uses `--no-deps`: Conda owns the binary dependency graph, while ordinary
Python tooling can use the dependency metadata in `pyproject.toml`.

## Local quality checks

```powershell
& .\.conda-env\python.exe -m pip check
& .\.conda-env\python.exe -m pytest -q
& .\.conda-env\Scripts\ruff.exe check .
& .\.conda-env\python.exe -m compileall -q src scripts tests
```

The same dataset-independent checks run in continuous integration for pull
requests and for updates to `main`.

## Artifact policy

Compact metrics and human-review tables are versioned. Large deterministic
inventories and all pixel-bearing evidence remain local. See
[artifacts/README.md](artifacts/README.md) for regeneration and checksum rules.

## Repository layout

- `src/fingerprint_new_method/` — reusable project code and configuration.
- `scripts/` — experiment and maintenance entry points.
- `tests/` — fast, dataset-independent regression tests.
- `docs/experiments/` — protocols and experiment reports.
- `artifacts/` — compact tracked evidence plus manifests for local outputs.

## Citation and rights

Repository citation metadata is provided in `CITATION.cff`. Repository-owned
content is covered by `LICENSE`; external datasets retain their own terms.
