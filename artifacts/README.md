# Artifact policy

Artifacts are divided into three classes:

- **Tracked evidence:** compact CSV and JSON files needed to support the
  experiment reports, including human review decisions and aggregate metrics.
- **Regenerable large outputs:** deterministic tables that are excluded from
  Git and represented by a checksum manifest.
- **Local pixel evidence:** overlays, crops, and plates derived from external
  fingerprint images. These remain under `evidence-pixels/` and are never
  tracked.

The excluded Experiment 002 annotation inventory is regenerated with:

```powershell
& .\.conda-env\python.exe .\scripts\experiment_002_inventory_and_sample.py
```

Its expected size, row count, and checksum are recorded in
`experiment-002/inventory/annotations.manifest.json`.

Experiment 003 keeps its two-channel impression score matrices under the
ignored `experiment-003/local-large/` directory. Regenerate them and their
compact checksum manifest with:

```powershell
& .\.conda-env\python.exe .\scripts\experiment_003_build_crosswalk.py --overwrite
```

Its blinded comparison plates remain under the ignored
`experiment-003/evidence-pixels/crosswalk-review/` directory and are regenerated
with `scripts/experiment_003_prepare_crosswalk_review.py`.

Experiment 004 keeps preprocessed images, checkpoints, heatmaps, and dense
registration maps under the ignored `experiment-004/local-large/` directory.
Compact SHA-256 manifests and metrics remain tracked. The frozen sequence is:

```powershell
& .\.conda-env\python.exe .\scripts\experiment_004_prepare.py
& .\.conda-env\python.exe .\scripts\experiment_004_preprocess_cache.py
& .\.conda-env\python.exe .\scripts\experiment_004_train.py
& .\.conda-env\python.exe .\scripts\experiment_004_infer_l3sf.py --partition validation
& .\.conda-env\python.exe .\scripts\experiment_004_freeze_and_score_l3sf.py --phase validation-freeze
& .\.conda-env\python.exe .\scripts\experiment_004_preprocess_cache.py --partition test
& .\.conda-env\python.exe .\scripts\experiment_004_infer_l3sf.py --partition test
& .\.conda-env\python.exe .\scripts\experiment_004_freeze_and_score_l3sf.py --phase test-score
```

Training writes an epoch-boundary state to `checkpoints/seed-<seed>/last.pt` and
resumes from it automatically, so an interrupted run continues the same
deterministic trajectory instead of restarting. The file is removed when the seed
finishes. Detached runs append to `local-large/logs/training.log`.

A long detached run must be started **without a console window**: the numerical
stack ships an Intel Fortran runtime that aborts the process on a console
control event, so a visible console someone closes ends the run with
`forrtl: error (200)`. Launch it with `DETACHED_PROCESS` and
`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`, as `local-large/run_training.cmd` does:

```powershell
$startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$startup.CreateFlags = 8
([wmiclass]"Win32_Process").Create(
  'cmd.exe /c "C:\fingerprint-new-method\artifacts\experiment-004\local-large\run_training.cmd"',
  'C:\fingerprint-new-method',
  $startup
)
```

SD300 commands are permitted only when Gate A is not `FAIL`; the 1000-ppi
outputs must be frozen before the 2000-ppi sensitivity run, and Experiment 001
ratings are opened only by `experiment_004_finalize.py` after both are frozen.
The SD300 ridge-scale step materializes the wide band of
`docs/experiments/004-scale-guard-contingency-amendment.md` once and tags each
image with its status under both bands; the frozen `[0.20, 1.50]` band remains
the primary analysis and alone decides Gate B.

See `docs/data-and-licensing.md` before adding or publishing any new artifact.
