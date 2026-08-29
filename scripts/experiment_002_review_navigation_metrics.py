#!/usr/bin/env python3
"""Compute review-navigation measurements for Experiment 002.

These measurements are deliberately not a pore classifier and do not assign audit
labels.  They only provide a fixed ordering for a human reviewer to revisit
low-contrast or low-texture crops after the blind sample has been frozen.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fingerprint_new_method.paths import dataset_path

WORK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = dataset_path("L3_SF_V2", "L3SF_V2")
EXP_ROOT = WORK_ROOT / "artifacts" / "experiment-002"
MANIFEST_PATH = EXP_ROOT / "review" / "review_annotation_manifest.csv"
OUTPUT_CSV = EXP_ROOT / "review" / "review_navigation_metrics.csv"
OUTPUT_JSON = EXP_ROOT / "review" / "review_navigation_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def disk_values(gray: np.ndarray, x: int, y: int, r_min: float, r_max: float) -> np.ndarray:
    radius = int(np.ceil(r_max))
    x0, x1 = max(0, x - radius), min(gray.shape[1] - 1, x + radius)
    y0, y1 = max(0, y - radius), min(gray.shape[0] - 1, y + radius)
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask = (distance >= r_min) & (distance <= r_max)
    return gray[y0 : y1 + 1, x0 : x1 + 1][mask].astype(np.float64)


def local_values(gray: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    return gray[
        max(0, y - radius) : min(gray.shape[0], y + radius + 1),
        max(0, x - radius) : min(gray.shape[1], x + radius + 1),
    ].astype(np.float64)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    rows = read_csv(MANIFEST_PATH)
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1000 frozen annotations, found {len(rows)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["image_relative_path"]].append(row)

    output: list[dict[str, Any]] = []
    for relative_path, image_rows in grouped.items():
        image = cv2.imread(str(DATA_ROOT / relative_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot decode {DATA_ROOT / relative_path}")
        height, width = image.shape
        for row in image_rows:
            x, y = int(row["x"]), int(row["y"])
            center = disk_values(image, x, y, 0.0, 2.0)
            annulus = disk_values(image, x, y, 4.0, 8.0)
            local = local_values(image, x, y, 12)
            center_mean = float(center.mean())
            annulus_mean = float(annulus.mean())
            local_std = float(local.std())
            output.append(
                {
                    **row,
                    "edge_distance_px": min(x, y, width - 1 - x, height - 1 - y),
                    "center_disk_r2_mean_gray": center_mean,
                    "annulus_r4_r8_mean_gray": annulus_mean,
                    "center_minus_annulus_gray": center_mean - annulus_mean,
                    "local_r12_std_gray": local_std,
                    "navigation_only": True,
                }
            )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    contrast = [float(row["center_minus_annulus_gray"]) for row in output]
    texture = [float(row["local_r12_std_gray"]) for row in output]
    summary = {
        "status": "NAVIGATION_ONLY_NOT_A_CLASSIFIER",
        "frozen_annotation_count": len(output),
        "method": {
            "center": "mean grayscale in Euclidean disk radius <= 2 px",
            "context": "mean grayscale in Euclidean annulus radius 4..8 px",
            "texture": "grayscale standard deviation in clipped 25x25 px neighborhood",
            "use": "fixed post-freeze ordering for human visual reinspection only",
        },
        "center_minus_annulus_gray_percentiles": {
            "p00": min(contrast),
            "p05": percentile(contrast, 5),
            "p50": percentile(contrast, 50),
            "p95": percentile(contrast, 95),
            "p100": max(contrast),
        },
        "local_r12_std_gray_percentiles": {
            "p00": min(texture),
            "p05": percentile(texture, 5),
            "p50": percentile(texture, 50),
            "p95": percentile(texture, 95),
            "p100": max(texture),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(output)} navigation rows to {OUTPUT_CSV}")
    print(f"Wrote navigation summary to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
