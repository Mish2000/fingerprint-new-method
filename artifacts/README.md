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

SD300 commands are permitted only when Gate A is not `FAIL`; the 1000-ppi
outputs must be frozen before the 2000-ppi sensitivity run, and Experiment 001
ratings are opened only by `experiment_004_finalize.py` after both are frozen.

See `docs/data-and-licensing.md` before adding or publishing any new artifact.
