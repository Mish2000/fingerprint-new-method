#!/usr/bin/env python3
"""Check the empirical coordinate convention of L3-SF pore TSV records.

This is a descriptive registration check, not a detector: it measures mean
source-pixel brightness around every tabulated point under small fixed shifts.
Synthetic pore openings are visually lighter than their ridge context, so the
aggregate peak helps distinguish direct (x=column, y=row) use from a one-pixel
origin correction.  Axis swapping is checked separately.
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
ANNOTATIONS_PATH = WORK_ROOT / "artifacts" / "experiment-002" / "inventory" / "annotations.csv"
OUTPUT_PATH = WORK_ROOT / "artifacts" / "experiment-002" / "inventory" / "coordinate_convention_check.json"
SHIFTS = list(range(-4, 5))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    annotations = read_csv(ANNOTATIONS_PATH)
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in annotations:
        grouped[row["image_relative_path"]].append((int(row["x"]), int(row["y"])))

    sums = {(dx, dy): 0.0 for dy in SHIFTS for dx in SHIFTS}
    counts = {(dx, dy): 0 for dy in SHIFTS for dx in SHIFTS}
    swapped_sum = 0.0
    swapped_count = 0

    for relative_path, points in grouped.items():
        gray = cv2.imread(str(DATA_ROOT / relative_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Cannot decode {DATA_ROOT / relative_path}")
        height, width = gray.shape
        coords = np.asarray(points, dtype=np.int32)
        x = coords[:, 0]
        y = coords[:, 1]
        for dy in SHIFTS:
            yy = y + dy
            for dx in SHIFTS:
                xx = x + dx
                valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                values = gray[yy[valid], xx[valid]].astype(np.float64)
                sums[(dx, dy)] += float(values.sum())
                counts[(dx, dy)] += int(values.size)

        swap_valid = (y >= 0) & (y < width) & (x >= 0) & (x < height)
        swap_values = gray[x[swap_valid], y[swap_valid]].astype(np.float64)
        swapped_sum += float(swap_values.sum())
        swapped_count += int(swap_values.size)

    means = {key: sums[key] / counts[key] for key in sums}
    best_shift = max(means, key=means.get)
    direct_mean = means[(0, 0)]
    minus_one_mean = means[(-1, -1)]
    swapped_mean = swapped_sum / swapped_count

    matrix: list[dict[str, Any]] = []
    for dy in SHIFTS:
        matrix.append(
            {
                "dy": dy,
                "mean_gray_by_dx": {str(dx): means[(dx, dy)] for dx in SHIFTS},
                "valid_count_by_dx": {str(dx): counts[(dx, dy)] for dx in SHIFTS},
            }
        )

    result = {
        "experiment_id": "002-l3sf-pore-annotation-feasibility",
        "status": "DESCRIPTIVE_COORDINATE_REGISTRATION_CHECK",
        "annotation_count": len(annotations),
        "observed_coordinate_range": {
            "x_min": min(x for points in grouped.values() for x, _ in points),
            "x_max": max(x for points in grouped.values() for x, _ in points),
            "y_min": min(y for points in grouped.values() for _, y in points),
            "y_max": max(y for points in grouped.values() for _, y in points),
        },
        "method": "Mean grayscale at all tabulated points under each fixed integer shift dx,dy in [-4,4]; higher aggregate brightness is expected at light pore openings. No labels, fitting, or detector involved.",
        "best_shift_from_tabulated_xy": {"dx": best_shift[0], "dy": best_shift[1], "mean_gray": means[best_shift]},
        "direct_xy_mean_gray": direct_mean,
        "minus_one_xy_mean_gray": minus_one_mean,
        "direct_minus_minus_one_mean_gray": direct_mean - minus_one_mean,
        "axis_swapped_mean_gray": swapped_mean,
        "direct_minus_axis_swapped_mean_gray": direct_mean - swapped_mean,
        "interpretation_rule": "Direct use is empirically favored only if its aggregate response peaks at or effectively ties the fixed-shift maximum and exceeds both (-1,-1) and axis-swapped controls.",
        "shift_response_matrix": matrix,
        "scientific_role": "Coordinate convention confirmation only; not a pore-localization performance result.",
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Best shift: dx={best_shift[0]}, dy={best_shift[1]}, mean={means[best_shift]:.6f}")
    print(f"Direct: {direct_mean:.6f}; minus-one: {minus_one_mean:.6f}; swapped: {swapped_mean:.6f}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
