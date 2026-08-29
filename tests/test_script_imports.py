"""Ensure experiment entry points load without local dataset access."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from fingerprint_new_method.paths import PROJECT_ROOT

SCRIPT_NAMES = sorted(path.name for path in (PROJECT_ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_script_loads_without_running_experiment(script_name: str) -> None:
    script_path = PROJECT_ROOT / "scripts" / script_name
    runpy.run_path(str(script_path), run_name=f"_smoke_{Path(script_name).stem}")
