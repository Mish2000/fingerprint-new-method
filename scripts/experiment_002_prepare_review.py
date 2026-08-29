#!/usr/bin/env python3
"""Prepare pixel-faithful local review sheets for Experiment 002."""

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
SAMPLE_PATH = EXP_ROOT / "review" / "review_sample_manifest.json"
EVIDENCE_ROOT = EXP_ROOT / "evidence-pixels" / "blind-audit"
INDEX_PATH = EXP_ROOT / "review" / "review_evidence_index.csv"

CROP_RADIUS = 24
SCALE = 4
CELL_W = 208
CELL_H = 226
COLS = 5
ROWS = 4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_stem(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def crop_with_padding(image: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    size = 2 * radius
    result = np.full((size, size, 3), 127, dtype=np.uint8)
    x0, y0, x1, y1 = x - radius, y - radius, x + radius, y + radius
    source_x0, source_y0 = max(0, x0), max(0, y0)
    source_x1, source_y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    target_x0, target_y0 = source_x0 - x0, source_y0 - y0
    target_x1 = target_x0 + source_x1 - source_x0
    target_y1 = target_y0 + source_y1 - source_y0
    if source_x1 > source_x0 and source_y1 > source_y0:
        result[target_y0:target_y1, target_x0:target_x1] = image[source_y0:source_y1, source_x0:source_x1]
    return result


def main() -> None:
    annotations = read_csv(MANIFEST_PATH)
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    if sample["image_count"] != 50 or len(annotations) != 1000:
        raise RuntimeError("Expected frozen sample of 50 images and 1000 annotations")

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in annotations:
        grouped[int(row["audit_image_index"])].append(row)

    evidence_index: list[dict[str, Any]] = []
    for image_entry in sample["selected_images"]:
        audit_index = int(image_entry["audit_image_index"])
        rows = sorted(grouped[audit_index], key=lambda row: int(row["audit_annotation_index"]))
        if len(rows) != int(image_entry["selected_annotation_count"]):
            raise RuntimeError(f"Manifest mismatch for audit image {audit_index}")
        image_path = DATA_ROOT / image_entry["image_relative_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode selected audit image: {image_path}")

        canonical = image_entry["canonical_image_id"]
        sample_name = f"S{audit_index:02}_{safe_stem(canonical)}"
        output_dir = EVIDENCE_ROOT / sample_name
        crop_dir = output_dir / "raw-crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        overview = image.copy()
        sheet = np.full((ROWS * CELL_H, COLS * CELL_W, 3), 245, dtype=np.uint8)
        for row in rows:
            annotation_index = int(row["audit_annotation_index"])
            source_row = int(row["annotation_row_index"])
            x, y = int(row["x"]), int(row["y"])
            raw_crop = crop_with_padding(image, x, y, CROP_RADIUS)
            crop_path = crop_dir / f"A{annotation_index:02}_row{source_row:04}_x{x}_y{y}.png"
            if not cv2.imwrite(str(crop_path), raw_crop):
                raise RuntimeError(f"Failed to write {crop_path}")

            display = cv2.resize(raw_crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
            center = (CROP_RADIUS * SCALE, CROP_RADIUS * SCALE)
            cv2.circle(display, center, 7 * SCALE, (0, 0, 255), 2, cv2.LINE_AA)
            tick_inner, tick_outer = 9 * SCALE, 12 * SCALE
            cv2.line(display, (center[0] - tick_outer, center[1]), (center[0] - tick_inner, center[1]), (0, 0, 255), 2)
            cv2.line(display, (center[0] + tick_inner, center[1]), (center[0] + tick_outer, center[1]), (0, 0, 255), 2)
            cv2.line(display, (center[0], center[1] - tick_outer), (center[0], center[1] - tick_inner), (0, 0, 255), 2)
            cv2.line(display, (center[0], center[1] + tick_inner), (center[0], center[1] + tick_outer), (0, 0, 255), 2)

            grid_row = (annotation_index - 1) // COLS
            grid_col = (annotation_index - 1) % COLS
            cell_x, cell_y = grid_col * CELL_W, grid_row * CELL_H
            sheet[cell_y + 28 : cell_y + 28 + display.shape[0], cell_x + 8 : cell_x + 8 + display.shape[1]] = display
            label = f"A{annotation_index:02} row={source_row} ({x},{y})"
            cv2.putText(sheet, label, (cell_x + 7, cell_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

            cv2.circle(overview, (x, y), 7, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(overview, str(annotation_index), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 255), 1, cv2.LINE_AA)

            evidence_index.append(
                {
                    **row,
                    "raw_crop_relative_path": crop_path.relative_to(WORK_ROOT).as_posix(),
                    "crop_source_box_xyxy": f"[{x-CROP_RADIUS},{y-CROP_RADIUS},{x+CROP_RADIUS},{y+CROP_RADIUS}]",
                    "display_interpolation": "INTER_NEAREST_4X",
                    "marker_status": "DISPLAY_ONLY_RED_RING_CENTER_PIXEL_UNOCCLUDED",
                }
            )

        title_bar = np.full((42, sheet.shape[1], 3), 255, dtype=np.uint8)
        title = f"S{audit_index:02} {canonical} | red ring=TSV (x,y); nearest-neighbor display; classify from source pixels"
        cv2.putText(title_bar, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 1, cv2.LINE_AA)
        final_sheet = np.vstack([title_bar, sheet])
        sheet_path = output_dir / "review_sheet.png"
        overview_path = output_dir / "overview_selected_annotations.png"
        if not cv2.imwrite(str(sheet_path), final_sheet) or not cv2.imwrite(str(overview_path), overview):
            raise RuntimeError(f"Failed writing review evidence for {canonical}")

    fields = list(evidence_index[0])
    with INDEX_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(evidence_index)

    print(f"Prepared {len(grouped)} review sheets and {len(evidence_index)} raw crops")
    print(f"Evidence root: {EVIDENCE_ROOT}")
    print(f"Index: {INDEX_PATH}")


if __name__ == "__main__":
    main()
