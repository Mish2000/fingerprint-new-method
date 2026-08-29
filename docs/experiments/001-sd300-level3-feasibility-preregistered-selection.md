# Experiment 001 — SD300 Level-3 feasibility: frozen selection protocol

This file records the sampling procedure applied before any SD300 image pixels
were decoded or inspected for quality.

- Experiment: `001-sd300-level3-feasibility`
- Fixed seed: `nist-sd300-level3-exp001-v1`
- Sampling frame: subjects for which all six expected PNG files (plain and roll
  at 500, 1000, and 2000 ppi) exist and are listed in the corresponding NIST
  checksum manifests.
- Assignment order: anatomical positions 01 through 10, ascending.
- Ranking: `SHA256(seed|two-digit anatomical position|subject ID)`, ascending.
- Selection: the first two integrity-valid candidates per position, with no
  subject reused at another position.
- Integrity validation: actual SHA-256 of each of the six selected files must
  equal the NIST-distributed checksum. A missing file, missing official checksum,
  unreadable or decoded-corrupt source, or checksum mismatch is an objective
  exclusion; the next ranked candidate is then considered.
- Quality blindness: filenames, availability, checksum metadata, and raw bytes
  for hashing may be read before selection is frozen; image pixels may not be
  decoded or viewed.
- Plain/anatomical mapping: FRGP 11 → right thumb (01), 12 → left thumb (06),
  and FRGP 02–05/07–10 map directly to the same anatomical positions.

The machine-readable frozen result is written to
`artifacts/experiment-001/selection/selection_manifest.json`, with a flat CSV
copy alongside it. Any objective exclusions and deterministic replacements are
recorded in the JSON manifest.

## Objective source exclusion after the initial freeze

After the initial selection was frozen and image pixels were first decoded,
`00001724` / anatomical position 05 was found to contain an objectively corrupt
plain-2000 source payload: the 500/1000 files show the fingerprint, while the
officially checksummed 2000 file decodes as horizontal scan-line noise above a
large black field. The initial manifests are retained as
`selection_manifest_initial.json` and `selection_manifest_initial.csv`. Under
the pre-registered rule for corrupt source data, the next eligible, unused,
integrity-valid subject in the already-fixed SHA-256 ranking replaces it. The
replacement candidate is selected and hash-verified before its pixels are
decoded.
