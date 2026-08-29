# Data and licensing policy

## Repository boundary

This repository contains source code, experiment protocols, compact metrics,
human review tables, and reproducibility metadata. It does not distribute the
source fingerprint images. Pixel-bearing evidence and large deterministic
inventories remain local and are excluded by `.gitignore`.

The dataset tree is an external, read-only input. Scripts resolve it from the
`FINGERPRINT_DATASETS_ROOT` environment variable. When that variable is not
set, they use a sibling directory named `fingerprint-datasets` next to the
repository checkout.

## NIST Special Database 300

NIST SD300 is used only as an external research input. Its catalog entry states
that the database must not be further distributed, published, copied, or
disseminated. No SD300 image is tracked in this repository.

- Dataset publication: <https://doi.org/10.6028/NIST.TN.1993>
- Catalog and terms: <https://catalog.data.gov/dataset/nist-special-database-300-uncompressed-plain-and-rolled-images-from-fingerprint-cards>

## L3-SF

L3-SF is a synthetic fingerprint dataset that includes pore annotations. The
project page identifies the work as licensed under CC BY-NC-SA 4.0. Source
images and the complete generated annotation inventory are not tracked here;
compact derived results retain dataset attribution and must not be interpreted
as relicensing the source material.

- Project and license: <https://andrewyzy.github.io/L3-SF/>
- Paper: <https://arxiv.org/abs/2002.03809>

## Publication rule

Before changing repository visibility or publishing any new artifact, verify
all of the following:

1. The artifact contains no source pixels or reconstructable image payload.
2. Redistribution is permitted by the applicable dataset terms.
3. Required attribution and share-alike obligations are preserved.
4. Local absolute paths, credentials, and private identifiers are absent.
5. Large deterministic outputs are represented by a compact manifest instead
   of normal Git history unless a deliberate LFS or release policy is adopted.

The repository's `LICENSE` applies only to repository-owned content. Dataset
terms remain separate and controlling for their respective materials.
