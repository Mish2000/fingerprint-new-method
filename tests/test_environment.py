"""Fast smoke tests for the project runtime."""

from __future__ import annotations

import sys

import cv2
import numpy as np
from PIL import Image

import fingerprint_new_method


def test_supported_python_version() -> None:
    assert sys.version_info[:2] == (3, 12)


def test_project_package_is_editable_and_importable() -> None:
    assert fingerprint_new_method.__version__ == "0.1.0"


def test_opencv_numpy_round_trip() -> None:
    source = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    encoded, payload = cv2.imencode(".png", source)
    assert encoded
    decoded = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(decoded, source)


def test_opencv_build_is_headless() -> None:
    build_lines = (line.strip() for line in cv2.getBuildInformation().splitlines())
    gui_line = next(line for line in build_lines if line.startswith("GUI:"))
    assert gui_line.split(":", maxsplit=1)[1].strip() == "NONE"


def test_pillow_numpy_interoperability() -> None:
    source = np.zeros((4, 5, 3), dtype=np.uint8)
    image = Image.fromarray(source, mode="RGB")
    assert image.size == (5, 4)
