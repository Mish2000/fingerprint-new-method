"""Regression tests for portable external-dataset path resolution."""

from __future__ import annotations

from pathlib import Path

from fingerprint_new_method.paths import DATASETS_ROOT_ENV, PROJECT_ROOT, dataset_path, datasets_root


def test_default_dataset_root_is_a_sibling_of_the_checkout(monkeypatch) -> None:
    monkeypatch.delenv(DATASETS_ROOT_ENV, raising=False)
    assert datasets_root() == (PROJECT_ROOT.parent / "fingerprint-datasets").resolve()


def test_environment_override_controls_dataset_paths(monkeypatch, tmp_path: Path) -> None:
    configured_root = tmp_path / "external-data"
    monkeypatch.setenv(DATASETS_ROOT_ENV, str(configured_root))
    assert datasets_root() == configured_root.resolve()
    assert dataset_path("NIST") == configured_root.resolve() / "NIST"
