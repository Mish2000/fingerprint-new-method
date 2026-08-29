"""Portable project and external-dataset path resolution."""

from __future__ import annotations

import os
from pathlib import Path

DATASETS_ROOT_ENV = "FINGERPRINT_DATASETS_ROOT"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def datasets_root() -> Path:
    """Return the configured read-only dataset root without requiring it to exist."""
    configured_root = os.environ.get(DATASETS_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return (PROJECT_ROOT.parent / "fingerprint-datasets").resolve()


def dataset_path(*relative_parts: str) -> Path:
    """Resolve a path below the configured external dataset root."""
    return datasets_root().joinpath(*relative_parts)
