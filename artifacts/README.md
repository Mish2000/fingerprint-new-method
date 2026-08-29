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

See `docs/data-and-licensing.md` before adding or publishing any new artifact.
