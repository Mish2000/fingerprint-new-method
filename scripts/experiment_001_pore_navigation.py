#!/usr/bin/env python3
"""Mark repeatable small ridge voids as a navigation aid, never ground truth.

Inputs are the previously generated cross-impression plates. The upper-left
panel is an unmodified 2000 ppi plain crop; the lower-left panel is the aligned
roll display. Detections only direct visual review back to the raw source crops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts" / "experiment-001" / "evidence-pixels"
ANALYSIS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "experiment-001"
    / "analysis"
    / "pore_navigation_metrics.json"
)
PANEL_SIZE = 640
HEADER_SIZE = 34
CELL_HEIGHT = PANEL_SIZE + HEADER_SIZE


def extract_panels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    plate = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if plate is None or plate.shape[1] != PANEL_SIZE * 2 or plate.shape[0] != CELL_HEIGHT * 2:
        raise ValueError(f"Unexpected plate dimensions for {path}: {None if plate is None else plate.shape}")
    plain = plate[HEADER_SIZE:CELL_HEIGHT, 0:PANEL_SIZE].copy()
    aligned_roll = plate[
        CELL_HEIGHT + HEADER_SIZE : CELL_HEIGHT * 2, 0:PANEL_SIZE
    ].copy()
    return plain, aligned_roll


def normalize(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, (1.0, 99.0))
    if high - low < 8:
        return gray.copy()
    return np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(
        np.uint8
    )


def candidates(gray: np.ndarray) -> list[dict[str, float]]:
    display = normalize(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    opened = cv2.morphologyEx(display, cv2.MORPH_OPEN, kernel)
    top_hat = cv2.subtract(display, opened)
    context = cv2.GaussianBlur(display, (0, 0), 7.0)
    ridge_context = context < 205
    values = top_hat[ridge_context]
    threshold = max(12.0, float(np.percentile(values, 96.5))) if values.size else 12.0
    binary = np.logical_and(top_hat >= threshold, ridge_context).astype(np.uint8) * 255
    binary[:24, :] = 0
    binary[-24:, :] = 0
    binary[:, :24] = 0
    binary[:, -24:] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    result: list[dict[str, float]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if not (3 <= area <= 170 and width <= 24 and height <= 24):
            continue
        aspect = min(width, height) / max(width, height)
        if aspect < 0.24:
            continue
        cx, cy = centroids[label]
        ix, iy = int(round(cx)), int(round(cy))
        if not (24 <= ix < PANEL_SIZE - 24 and 24 <= iy < PANEL_SIZE - 24):
            continue
        yy, xx = np.ogrid[-14:15, -14:15]
        center_mask = xx * xx + yy * yy <= 4 * 4
        annulus_mask = np.logical_and(xx * xx + yy * yy >= 8 * 8, xx * xx + yy * yy <= 14 * 14)
        patch = display[iy - 14 : iy + 15, ix - 14 : ix + 15]
        if patch.shape != (29, 29):
            continue
        center_mean = float(patch[center_mask].mean())
        annulus_mean = float(patch[annulus_mask].mean())
        local_contrast = center_mean - annulus_mean
        if local_contrast < 7.0:
            continue
        strength = float(top_hat[labels == label].mean())
        result.append(
            {
                "x": float(cx),
                "y": float(cy),
                "area": float(area),
                "aspect": float(aspect),
                "local_contrast": local_contrast,
                "strength": strength,
                "score": strength + local_contrast * 0.7 + aspect * 4.0,
            }
        )
    result.sort(key=lambda item: (-item["score"], item["y"], item["x"]))
    return result


def mutual_spatial_matches(
    plain: list[dict[str, float]], roll: list[dict[str, float]], radius: float = 13.0
) -> list[dict[str, Any]]:
    if not plain or not roll:
        return []
    plain_points = np.asarray([[item["x"], item["y"]] for item in plain])
    roll_points = np.asarray([[item["x"], item["y"]] for item in roll])
    distances = np.linalg.norm(
        plain_points[:, None, :] - roll_points[None, :, :], axis=2
    )
    nearest_roll = distances.argmin(axis=1)
    nearest_plain = distances.argmin(axis=0)
    matches: list[dict[str, Any]] = []
    for plain_index, roll_index in enumerate(nearest_roll):
        if nearest_plain[roll_index] != plain_index:
            continue
        distance = float(distances[plain_index, roll_index])
        if distance > radius:
            continue
        score = min(plain[plain_index]["score"], roll[roll_index]["score"]) - distance
        matches.append(
            {
                "plain_index": int(plain_index),
                "roll_index": int(roll_index),
                "distance_px": distance,
                "score": score,
                "plain": plain[plain_index],
                "aligned_roll": roll[roll_index],
            }
        )
    matches.sort(key=lambda item: (-item["score"], item["distance_px"]))
    return matches


def best_local_cluster(matches: list[dict[str, Any]], radius: float = 175.0) -> list[dict[str, Any]]:
    if not matches:
        return []
    best: list[dict[str, Any]] = []
    for anchor in matches:
        anchor_point = np.asarray([anchor["plain"]["x"], anchor["plain"]["y"]])
        cluster = [
            match
            for match in matches
            if np.linalg.norm(
                np.asarray([match["plain"]["x"], match["plain"]["y"]])
                - anchor_point
            )
            <= radius
        ]
        cluster.sort(key=lambda item: (-item["score"], item["distance_px"]))
        cluster = cluster[:12]
        if (len(cluster), sum(item["score"] for item in cluster)) > (
            len(best),
            sum(item["score"] for item in best),
        ):
            best = cluster
    return best


def annotate(
    plain: np.ndarray,
    aligned_roll: np.ndarray,
    cluster: list[dict[str, Any]],
    output: Path,
) -> None:
    left = Image.fromarray(plain, mode="L").convert("RGB")
    right = Image.fromarray(aligned_roll, mode="L").convert("RGB")
    canvas = Image.new("RGB", (PANEL_SIZE * 2, PANEL_SIZE + 42), "white")
    canvas.paste(left, (0, 42))
    canvas.paste(right, (PANEL_SIZE, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "plain 2000 RAW — detector marks are navigation only", fill="black")
    draw.text(
        (PANEL_SIZE + 8, 10),
        "aligned roll display — verify against raw roll crop",
        fill="black",
    )
    for number, match in enumerate(cluster, start=1):
        color = (230, 20, 20)
        for offset, key in ((0, "plain"), (PANEL_SIZE, "aligned_roll")):
            x = float(match[key]["x"]) + offset
            y = float(match[key]["y"]) + 42
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=color, width=2)
            draw.text((x + 10, y - 11), str(number), fill=color)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    results: list[dict[str, Any]] = []
    for sample_dir in sorted(EVIDENCE_ROOT.glob("S??_*")):
        if not sample_dir.is_dir():
            continue
        for plate_path in sorted((sample_dir / "plates").glob("patch_*_cross_impression.png")):
            plain, aligned_roll = extract_panels(plate_path)
            plain_candidates = candidates(plain)
            roll_candidates = candidates(aligned_roll)
            matches = mutual_spatial_matches(plain_candidates, roll_candidates)
            cluster = best_local_cluster(matches)
            output = sample_dir / "pore-navigation" / plate_path.name.replace(
                "_cross_impression", "_candidate_voids"
            )
            annotate(plain, aligned_roll, cluster, output)
            results.append(
                {
                    "sample_key": sample_dir.name,
                    "patch": plate_path.stem.split("_cross_impression")[0],
                    "plain_candidate_count": len(plain_candidates),
                    "aligned_roll_candidate_count": len(roll_candidates),
                    "mutual_spatial_match_count": len(matches),
                    "best_local_cluster_count": len(cluster),
                    "best_local_cluster": cluster,
                    "annotated_plate": str(output.relative_to(EVIDENCE_ROOT)).replace("\\", "/"),
                    "scientific_status": "NAVIGATION_ONLY_NOT_GROUND_TRUTH",
                }
            )
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ANALYSIS_PATH}")
    for result in sorted(
        results,
        key=lambda item: (-item["best_local_cluster_count"], item["sample_key"], item["patch"]),
    )[:20]:
        print(
            f"{result['sample_key']} {result['patch']}: "
            f"cluster={result['best_local_cluster_count']} "
            f"matches={result['mutual_spatial_match_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
