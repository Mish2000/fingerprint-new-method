#!/usr/bin/env python3
"""Compute descriptive same-card cross-resolution metrics for experiment 001."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fingerprint_new_method.paths import dataset_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = dataset_path("NIST")
SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "experiment-001"
    / "selection"
    / "selection_manifest.json"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiment-001" / "analysis"


def read_gray(source_id: str) -> np.ndarray:
    image = cv2.imread(str(DATASET_ROOT / source_id), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not decode {source_id}")
    return image


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        left = a[mask].astype(np.float64)
        right = b[mask].astype(np.float64)
    else:
        left = a.astype(np.float64).ravel()
        right = b.astype(np.float64).ravel()
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def metrics(high: np.ndarray, low: np.ndarray) -> dict[str, Any]:
    expected_shape = (low.shape[0] * 2, low.shape[1] * 2)
    exact_2x = high.shape == expected_shape
    resized = cv2.resize(
        high,
        (low.shape[1], low.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    dark_union = np.logical_or(low < 210, resized < 210)
    difference = np.abs(low.astype(np.float32) - resized.astype(np.float32))
    return {
        "exact_2x_dimensions": exact_2x,
        "low_shape_hw": list(low.shape),
        "high_shape_hw": list(high.shape),
        "pearson_all_pixels": correlation(low, resized),
        "pearson_dark_union": correlation(low, resized, dark_union),
        "mae_all_pixels": float(difference.mean()),
        "mae_dark_union": float(difference[dark_union].mean()),
        "dark_union_pixel_fraction": float(dark_union.mean()),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for record in selection["selected"]:
        sources = {
            (source["impression"], source["ppi"]): source
            for source in record["sources"]
        }
        for impression in ("plain", "roll"):
            image_500 = read_gray(sources[(impression, 500)]["source_id"])
            image_1000 = read_gray(sources[(impression, 1000)]["source_id"])
            image_2000 = read_gray(sources[(impression, 2000)]["source_id"])
            values_1000_2000 = metrics(image_2000, image_1000)
            # The 500 ppi image is a descriptive low-resolution control only.
            image_1000_to_500 = cv2.resize(
                image_1000,
                (image_500.shape[1], image_500.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            dark_union_500 = np.logical_or(image_500 < 210, image_1000_to_500 < 210)
            rows.append(
                {
                    "sample_index": record["sample_index"],
                    "subject_id": record["subject_id"],
                    "anatomical_position": record["anatomical_position"],
                    "finger_name": record["finger_name"],
                    "impression": impression,
                    "source_500": sources[(impression, 500)]["source_id"],
                    "source_1000": sources[(impression, 1000)]["source_id"],
                    "source_2000": sources[(impression, 2000)]["source_id"],
                    "sha256_500": sources[(impression, 500)]["actual_sha256"],
                    "sha256_1000": sources[(impression, 1000)]["actual_sha256"],
                    "sha256_2000": sources[(impression, 2000)]["actual_sha256"],
                    **values_1000_2000,
                    "control_500_1000_pearson_all_pixels": correlation(
                        image_500, image_1000_to_500
                    ),
                    "control_500_1000_pearson_dark_union": correlation(
                        image_500, image_1000_to_500, dark_union_500
                    ),
                }
            )

    json_path = OUTPUT_DIR / "cross_resolution_metrics.json"
    csv_path = OUTPUT_DIR / "cross_resolution_metrics.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    correlations = [row["pearson_dark_union"] for row in rows]
    print(
        "1000<->2000 dark-union Pearson: "
        f"min={min(correlations):.6f} median={np.median(correlations):.6f} "
        f"max={max(correlations):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
