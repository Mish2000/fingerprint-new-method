#!/usr/bin/env python3
"""Prepare local, non-generative visual evidence for experiment 001.

The script uses 500 ppi ridge-level SIFT correspondences only to navigate between
plain and rolled impressions. Scientific judgments must be made from the saved
raw 2000 ppi source crops (and their exact 1000 ppi counterparts), not from SIFT
scores or interpolated aligned displays.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

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
EVIDENCE_ROOT = (
    PROJECT_ROOT / "artifacts" / "experiment-001" / "evidence-pixels"
)
ANALYSIS_ROOT = PROJECT_ROOT / "artifacts" / "experiment-001" / "analysis"

PATCH_HALF_500 = 80
PATCH_SIZE_2000 = PATCH_HALF_500 * 2 * 4
MAX_PATCHES = 4


def load_gray(source_id: str) -> np.ndarray:
    path = DATASET_ROOT / source_id
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    return image


def source_map(record: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (source["impression"], source["ppi"]): source
        for source in record["sources"]
    }


def odd(value: int) -> int:
    return value if value % 2 else value + 1


def fingerprint_mask(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Find the large, ridge-dense region while suppressing thin card rules/text."""
    otsu_value, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    dark_threshold = int(np.clip(otsu_value + 18, 135, 210))
    dark = (gray < dark_threshold).astype(np.float32)
    density_kernel = odd(max(25, round(min(gray.shape) * 0.065)))
    density = cv2.boxFilter(
        dark,
        ddepth=-1,
        ksize=(density_kernel, density_kernel),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    dense = (density > 0.13).astype(np.uint8) * 255
    close_size = odd(max(17, round(min(gray.shape) * 0.04)))
    open_size = odd(max(7, round(min(gray.shape) * 0.012)))
    dense = cv2.morphologyEx(
        dense,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )
    dense = cv2.morphologyEx(
        dense,
        cv2.MORPH_OPEN,
        np.ones((open_size, open_size), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dense)
    if count <= 1:
        mask = np.full_like(gray, 255, dtype=np.uint8)
        return mask, (0, 0, gray.shape[1], gray.shape[0])

    image_area = gray.shape[0] * gray.shape[1]
    choices: list[tuple[float, int]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < image_area * 0.01:
            continue
        compact_span = math.sqrt(max(width * height, 1))
        score = float(area) + compact_span * 5.0
        choices.append((score, label))
    if not choices:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        label = max(choices)[1]

    mask = (labels == label).astype(np.uint8) * 255
    dilation = odd(max(13, round(min(gray.shape) * 0.025)))
    mask = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))
    points = cv2.findNonZero(mask)
    if points is None:
        return np.full_like(gray, 255), (0, 0, gray.shape[1], gray.shape[0])
    bbox = cv2.boundingRect(points)
    return mask, tuple(int(value) for value in bbox)


def sift_features(
    gray: np.ndarray, mask: np.ndarray
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    display = clahe.apply(gray)
    sift = cv2.SIFT_create(
        nfeatures=12000,
        contrastThreshold=0.012,
        edgeThreshold=18,
        sigma=1.4,
    )
    return sift.detectAndCompute(display, mask)


def mutual_ratio_matches(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    ratio: float,
) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward_knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    reverse_knn = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    forward = {
        (pair[0].queryIdx, pair[0].trainIdx): pair[0]
        for pair in forward_knn
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    reverse = {
        (pair[0].trainIdx, pair[0].queryIdx)
        for pair in reverse_knn
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    return [match for key, match in forward.items() if key in reverse]


def align_plain_roll(
    plain: np.ndarray, roll: np.ndarray
) -> dict[str, Any]:
    plain_mask, plain_bbox = fingerprint_mask(plain)
    roll_mask, roll_bbox = fingerprint_mask(roll)
    plain_kp, plain_desc = sift_features(plain, plain_mask)
    roll_kp, roll_desc = sift_features(roll, roll_mask)
    if plain_desc is None or roll_desc is None:
        raise RuntimeError("SIFT did not find descriptors")

    matches: list[cv2.DMatch] = []
    used_ratio = 0.0
    for ratio in (0.70, 0.75, 0.80, 0.85):
        matches = mutual_ratio_matches(plain_desc, roll_desc, ratio)
        used_ratio = ratio
        if len(matches) >= 16:
            break
    if len(matches) < 4:
        raise RuntimeError(f"Only {len(matches)} mutual SIFT matches")

    source_points = np.float32([plain_kp[m.queryIdx].pt for m in matches])
    target_points = np.float32([roll_kp[m.trainIdx].pt for m in matches])
    homography, inlier_mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.USAC_MAGSAC,
        ransacReprojThreshold=5.0,
        maxIters=20000,
        confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed")
    inlier_flags = inlier_mask.ravel().astype(bool)
    inliers = [match for match, keep in zip(matches, inlier_flags) if keep]
    if len(inliers) < 4:
        raise RuntimeError(f"Only {len(inliers)} geometric inliers")

    # Once a conservative transform exists, admit additional mutual matches only
    # when they agree with it geometrically. These points broaden spatial coverage
    # for navigation; they are not interpreted as Level-3 evidence.
    relaxed_matches = mutual_ratio_matches(plain_desc, roll_desc, 0.92)
    relaxed_source = np.float32(
        [plain_kp[match.queryIdx].pt for match in relaxed_matches]
    )
    relaxed_target = np.float32(
        [roll_kp[match.trainIdx].pt for match in relaxed_matches]
    )
    if len(relaxed_matches):
        relaxed_projected = cv2.perspectiveTransform(
            relaxed_source.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        relaxed_errors = np.linalg.norm(relaxed_projected - relaxed_target, axis=1)
        navigation_matches = [
            match
            for match, error in zip(relaxed_matches, relaxed_errors)
            if error <= 9.0
        ]
    else:
        navigation_matches = []
    if len(navigation_matches) < len(inliers):
        navigation_matches = inliers

    projected = cv2.perspectiveTransform(
        source_points.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - target_points, axis=1)
    inlier_errors = errors[inlier_flags]
    inlier_plain_points = source_points[inlier_flags]
    inlier_roll_points = target_points[inlier_flags]

    def hull_fraction(points: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        if len(points) < 3:
            return 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(points.reshape(-1, 1, 2))))
        bbox_area = float(max(bbox[2] * bbox[3], 1))
        return hull_area / bbox_area

    warped_plain_mask = cv2.warpPerspective(
        plain_mask,
        homography,
        (roll.shape[1], roll.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_pixels = warped_plain_mask > 0
    mask_overlap = float(
        np.logical_and(warped_pixels, roll_mask > 0).sum()
        / max(int(warped_pixels.sum()), 1)
    )
    plain_coverage = hull_fraction(inlier_plain_points, plain_bbox)
    roll_coverage = hull_fraction(inlier_roll_points, roll_bbox)
    quality_score = (
        min(len(inliers), 100)
        + min(len(navigation_matches), 150) * 0.12
        + min(plain_coverage, roll_coverage) * 35.0
        + mask_overlap * 22.0
        - float(np.median(inlier_errors)) * 1.5
    )
    return {
        "plain_mask": plain_mask,
        "roll_mask": roll_mask,
        "plain_bbox": plain_bbox,
        "roll_bbox": roll_bbox,
        "plain_keypoints": plain_kp,
        "roll_keypoints": roll_kp,
        "all_matches": matches,
        "inliers": inliers,
        "navigation_matches": navigation_matches,
        "homography": homography,
        "ratio": used_ratio,
        "median_inlier_error_500": float(np.median(inlier_errors)),
        "p90_inlier_error_500": float(np.percentile(inlier_errors, 90)),
        "plain_inlier_hull_fraction": plain_coverage,
        "roll_inlier_hull_fraction": roll_coverage,
        "warped_plain_mask_overlap_fraction": mask_overlap,
        "navigation_quality_score": quality_score,
    }


def choose_navigation_alignment(
    images: dict[tuple[str, int], np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    diagnostics: list[dict[str, Any]] = []
    target_plain_shape = images[("plain", 500)].shape
    target_roll_shape = images[("roll", 500)].shape
    for source_ppi in (500, 1000, 2000):
        plain_source = images[("plain", source_ppi)]
        roll_source = images[("roll", source_ppi)]
        if source_ppi == 500:
            plain_navigation = plain_source
            roll_navigation = roll_source
        else:
            plain_navigation = cv2.resize(
                plain_source,
                (target_plain_shape[1], target_plain_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            roll_navigation = cv2.resize(
                roll_source,
                (target_roll_shape[1], target_roll_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        try:
            alignment = align_plain_roll(plain_navigation, roll_navigation)
            alignment["navigation_source_ppi"] = source_ppi
            candidates.append((plain_navigation, roll_navigation, alignment))
            diagnostics.append(
                {
                    "source_ppi": source_ppi,
                    "status": "SUCCESS",
                    "quality_score": alignment["navigation_quality_score"],
                    "geometric_inlier_count": len(alignment["inliers"]),
                    "navigation_match_count": len(alignment["navigation_matches"]),
                    "median_inlier_error_500_px": alignment[
                        "median_inlier_error_500"
                    ],
                    "plain_inlier_hull_fraction": alignment[
                        "plain_inlier_hull_fraction"
                    ],
                    "roll_inlier_hull_fraction": alignment[
                        "roll_inlier_hull_fraction"
                    ],
                    "warped_plain_mask_overlap_fraction": alignment[
                        "warped_plain_mask_overlap_fraction"
                    ],
                }
            )
        except Exception as error:
            diagnostics.append(
                {
                    "source_ppi": source_ppi,
                    "status": "FAILED",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if not candidates:
        raise RuntimeError(f"All navigation alignments failed: {diagnostics}")
    plain_navigation, roll_navigation, alignment = max(
        candidates,
        key=lambda item: (
            item[2]["navigation_quality_score"],
            len(item[2]["inliers"]),
            -item[2]["navigation_source_ppi"],
        ),
    )
    return plain_navigation, roll_navigation, alignment, diagnostics


def local_affine(
    alignment: dict[str, Any], center: np.ndarray
) -> np.ndarray:
    plain_kp = alignment["plain_keypoints"]
    roll_kp = alignment["roll_keypoints"]
    inliers = alignment["navigation_matches"]
    pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
    for match in inliers:
        source = np.asarray(plain_kp[match.queryIdx].pt, dtype=np.float32)
        target = np.asarray(roll_kp[match.trainIdx].pt, dtype=np.float32)
        distance = float(np.linalg.norm(source - center))
        if distance <= PATCH_HALF_500 * 2.75:
            pairs.append((source, target, distance))
    pairs.sort(key=lambda pair: pair[2])
    if len(pairs) >= 4:
        source_points = np.asarray([pair[0] for pair in pairs], dtype=np.float32)
        target_points = np.asarray([pair[1] for pair in pairs], dtype=np.float32)
        affine, mask = cv2.estimateAffine2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=10000,
            confidence=0.999,
            refineIters=20,
        )
        if affine is not None and mask is not None and int(mask.sum()) >= 3:
            return np.vstack([affine, [0.0, 0.0, 1.0]])
    return alignment["homography"].copy()


def choose_patch_matches(
    alignment: dict[str, Any], plain_shape: tuple[int, int], roll_shape: tuple[int, int]
) -> list[dict[str, Any]]:
    plain_kp = alignment["plain_keypoints"]
    roll_kp = alignment["roll_keypoints"]
    plain_mask = alignment["plain_mask"]
    roll_mask = alignment["roll_mask"]
    margin = PATCH_HALF_500 + 4
    candidates: list[tuple[float, cv2.DMatch, np.ndarray, np.ndarray]] = []
    for match in alignment["navigation_matches"]:
        source = np.asarray(plain_kp[match.queryIdx].pt, dtype=np.float32)
        target = np.asarray(roll_kp[match.trainIdx].pt, dtype=np.float32)
        sx, sy = int(round(source[0])), int(round(source[1]))
        tx, ty = int(round(target[0])), int(round(target[1]))
        if not (
            margin <= sx < plain_shape[1] - margin
            and margin <= sy < plain_shape[0] - margin
            and margin <= tx < roll_shape[1] - margin
            and margin <= ty < roll_shape[0] - margin
        ):
            continue
        if plain_mask[sy, sx] == 0 or roll_mask[ty, tx] == 0:
            continue
        response = min(
            plain_kp[match.queryIdx].response,
            roll_kp[match.trainIdx].response,
        )
        score = float(response) / (1.0 + match.distance / 150.0)
        candidates.append((score, match, source, target))
    candidates.sort(key=lambda item: (-item[0], item[1].queryIdx, item[1].trainIdx))

    chosen: list[dict[str, Any]] = []
    min_separation = PATCH_HALF_500 * 1.05
    for score, match, source, target in candidates:
        if any(
            np.linalg.norm(source - item["plain_center_500"]) < min_separation
            or np.linalg.norm(target - item["roll_center_500"]) < min_separation
            for item in chosen
        ):
            continue
        chosen.append(
            {
                "match": match,
                "navigation_score": score,
                "plain_center_500": source,
                "roll_center_500": target,
                "plain_to_roll_local": local_affine(alignment, source),
            }
        )
        if len(chosen) == MAX_PATCHES:
            break
    return chosen


def crop_exact(
    image: np.ndarray, center: tuple[float, float], size: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x = int(round(center[0]))
    y = int(round(center[1]))
    half = size // 2
    x0, y0, x1, y1 = x - half, y - half, x + half, y + half
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        raise ValueError(f"Crop {(x0, y0, x1, y1)} exceeds image {image.shape}")
    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def display_contrast(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, (1.0, 99.0))
    if high - low < 8:
        return gray.copy()
    return np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(
        np.uint8
    )


def label_panel(gray: np.ndarray, label: str) -> Image.Image:
    panel = Image.fromarray(gray, mode="L").convert("RGB")
    canvas = Image.new("RGB", (panel.width, panel.height + 34), "white")
    canvas.paste(panel, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill="black")
    return canvas


def concatenate_panels(panels: list[Image.Image]) -> Image.Image:
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    result = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        result.paste(panel, (x, 0))
        x += panel.width
    return result


def panel_grid_2x2(panels: list[Image.Image]) -> Image.Image:
    if len(panels) != 4:
        raise ValueError("panel_grid_2x2 requires exactly four panels")
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    result = Image.new("RGB", (cell_width * 2, cell_height * 2), "white")
    for index, panel in enumerate(panels):
        x = (index % 2) * cell_width
        y = (index // 2) * cell_height
        result.paste(panel, (x, y))
    return result


def refine_aligned_display(
    plain_crop: np.ndarray, initially_aligned_roll: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    full_mask = np.full_like(plain_crop, 255, dtype=np.uint8)
    plain_kp, plain_desc = sift_features(plain_crop, full_mask)
    roll_kp, roll_desc = sift_features(initially_aligned_roll, full_mask)
    if plain_desc is None or roll_desc is None:
        return initially_aligned_roll, {"status": "NO_DESCRIPTORS"}
    matches: list[cv2.DMatch] = []
    ratio_used = 0.0
    for ratio in (0.72, 0.78, 0.84, 0.90):
        matches = mutual_ratio_matches(plain_desc, roll_desc, ratio)
        ratio_used = ratio
        if len(matches) >= 10:
            break
    if len(matches) < 4:
        return initially_aligned_roll, {
            "status": "INSUFFICIENT_MATCHES",
            "mutual_match_count": len(matches),
            "ratio_threshold_used": ratio_used,
        }

    plain_points = np.float32([plain_kp[m.queryIdx].pt for m in matches])
    roll_points = np.float32([roll_kp[m.trainIdx].pt for m in matches])
    affine, inlier_mask = cv2.estimateAffine2D(
        roll_points,
        plain_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=7.0,
        maxIters=10000,
        confidence=0.999,
        refineIters=25,
    )
    if affine is None or inlier_mask is None or int(inlier_mask.sum()) < 4:
        return initially_aligned_roll, {
            "status": "AFFINE_FAILED",
            "mutual_match_count": len(matches),
            "ratio_threshold_used": ratio_used,
        }

    linear = affine[:, :2]
    determinant = float(np.linalg.det(linear))
    singular_values = np.linalg.svd(linear, compute_uv=False)
    translation = float(np.linalg.norm(affine[:, 2]))
    # Reject implausible corrections: the first-stage local transform already
    # handles the impression geometry, so refinement must stay modest.
    if not (
        0.72 <= determinant <= 1.38
        and singular_values.min() >= 0.72
        and singular_values.max() <= 1.38
        and translation <= 120.0
    ):
        return initially_aligned_roll, {
            "status": "IMPLAUSIBLE_CORRECTION_REJECTED",
            "mutual_match_count": len(matches),
            "inlier_count": int(inlier_mask.sum()),
            "determinant": determinant,
            "singular_values": singular_values.tolist(),
            "translation_px": translation,
        }
    refined = cv2.warpAffine(
        initially_aligned_roll,
        affine,
        (PATCH_SIZE_2000, PATCH_SIZE_2000),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return refined, {
        "status": "APPLIED_DISPLAY_ONLY",
        "mutual_match_count": len(matches),
        "inlier_count": int(inlier_mask.sum()),
        "ratio_threshold_used": ratio_used,
        "determinant": determinant,
        "singular_values": singular_values.tolist(),
        "translation_px": translation,
        "aligned_roll_to_plain_affine_2000": affine.tolist(),
    }


def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 64) -> np.ndarray:
    yy, xx = np.indices(a.shape)
    choose_a = ((xx // tile) + (yy // tile)) % 2 == 0
    return np.where(choose_a, a, b).astype(np.uint8)


def scaled_homography(matrix_500: np.ndarray, factor: float = 4.0) -> np.ndarray:
    up = np.diag([factor, factor, 1.0])
    down = np.diag([1.0 / factor, 1.0 / factor, 1.0])
    return up @ matrix_500 @ down


def aligned_roll_patch(
    roll_2000: np.ndarray,
    plain_center_2000: tuple[float, float],
    plain_to_roll_500: np.ndarray,
) -> np.ndarray:
    transform = scaled_homography(plain_to_roll_500)
    roll_to_plain = np.linalg.inv(transform)
    half = PATCH_SIZE_2000 // 2
    x0 = plain_center_2000[0] - half
    y0 = plain_center_2000[1] - half
    translate = np.asarray(
        [[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    source_to_patch = translate @ roll_to_plain
    return cv2.warpPerspective(
        roll_2000,
        source_to_patch,
        (PATCH_SIZE_2000, PATCH_SIZE_2000),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def save_match_diagnostic(
    sample_dir: Path,
    plain_500: np.ndarray,
    roll_500: np.ndarray,
    alignment: dict[str, Any],
) -> None:
    inliers = sorted(
        alignment["inliers"], key=lambda match: match.distance
    )[:80]
    diagnostic = cv2.drawMatches(
        plain_500,
        alignment["plain_keypoints"],
        roll_500,
        alignment["roll_keypoints"],
        inliers,
        None,
        matchColor=(0, 150, 0),
        singlePointColor=(160, 160, 160),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(sample_dir / "navigation_matches_500.png"), diagnostic)


def overview_panel(gray: np.ndarray, bbox: tuple[int, int, int, int], label: str) -> Image.Image:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    x, y, width, height = bbox
    cv2.rectangle(rgb, (x, y), (x + width, y + height), (220, 30, 30), 3)
    image = Image.fromarray(rgb)
    image.thumbnail((800, 650), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (800, 700), "white")
    canvas.paste(image, ((800 - image.width) // 2, 42))
    ImageDraw.Draw(canvas).text((10, 10), label, fill="black")
    return canvas


def process_record(record: dict[str, Any]) -> dict[str, Any]:
    sample_index = record["sample_index"]
    sample_key = f"S{sample_index:02d}_{record['subject_id']}_P{record['anatomical_position']:02d}"
    sample_dir = EVIDENCE_ROOT / sample_key
    raw_dir = sample_dir / "raw-crops"
    plate_dir = sample_dir / "plates"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plate_dir.mkdir(parents=True, exist_ok=True)

    sources = source_map(record)
    images = {
        key: load_gray(source["source_id"])
        for key, source in sources.items()
    }
    plain_500 = images[("plain", 500)]
    roll_500 = images[("roll", 500)]
    (
        plain_navigation,
        roll_navigation,
        alignment,
        alignment_candidates,
    ) = choose_navigation_alignment(images)
    patches = choose_patch_matches(
        alignment, plain_500.shape, roll_500.shape
    )
    save_match_diagnostic(sample_dir, plain_navigation, roll_navigation, alignment)

    overview = concatenate_panels(
        [
            overview_panel(
                images[("plain", 2000)],
                tuple(value * 4 for value in alignment["plain_bbox"]),
                "plain 2000 — red box is navigation foreground mask bbox",
            ),
            overview_panel(
                images[("roll", 2000)],
                tuple(value * 4 for value in alignment["roll_bbox"]),
                "roll 2000 — red box is navigation foreground mask bbox",
            ),
        ]
    )
    overview.save(sample_dir / "overview_2000.png")

    patch_records: list[dict[str, Any]] = []
    for patch_index, patch in enumerate(patches, start=1):
        plain_center_2000 = tuple(float(value * 4.0) for value in patch["plain_center_500"])
        roll_center_2000 = tuple(float(value * 4.0) for value in patch["roll_center_500"])
        plain_crop, plain_box = crop_exact(
            images[("plain", 2000)], plain_center_2000, PATCH_SIZE_2000
        )
        roll_crop, roll_box = crop_exact(
            images[("roll", 2000)], roll_center_2000, PATCH_SIZE_2000
        )
        plain_1000_crop, plain_1000_box = crop_exact(
            images[("plain", 1000)],
            (plain_center_2000[0] / 2.0, plain_center_2000[1] / 2.0),
            PATCH_SIZE_2000 // 2,
        )
        roll_1000_crop, roll_1000_box = crop_exact(
            images[("roll", 1000)],
            (roll_center_2000[0] / 2.0, roll_center_2000[1] / 2.0),
            PATCH_SIZE_2000 // 2,
        )
        aligned_roll_initial = aligned_roll_patch(
            images[("roll", 2000)],
            plain_center_2000,
            patch["plain_to_roll_local"],
        )
        aligned_roll, refinement_metrics = refine_aligned_display(
            plain_crop, aligned_roll_initial
        )

        prefix = f"patch_{patch_index:02d}"
        cv2.imwrite(str(raw_dir / f"{prefix}_plain_2000_raw.png"), plain_crop)
        cv2.imwrite(str(raw_dir / f"{prefix}_roll_2000_raw.png"), roll_crop)
        cv2.imwrite(str(raw_dir / f"{prefix}_plain_1000_raw.png"), plain_1000_crop)
        cv2.imwrite(str(raw_dir / f"{prefix}_roll_1000_raw.png"), roll_1000_crop)

        plain_display = display_contrast(plain_crop)
        aligned_display = display_contrast(aligned_roll)
        cross_impression_plate = panel_grid_2x2(
            [
                label_panel(plain_crop, "plain 2000 — RAW source crop"),
                label_panel(roll_crop, "roll 2000 — RAW source crop"),
                label_panel(
                    aligned_display,
                    "roll 2000 — aligned/contrast display only",
                ),
                label_panel(
                    checkerboard(plain_display, aligned_display),
                    "plain/roll checkerboard — aligned display only",
                ),
            ]
        )
        cross_impression_plate.save(plate_dir / f"{prefix}_cross_impression.png")

        plain_1000_up = cv2.resize(
            plain_1000_crop,
            (PATCH_SIZE_2000, PATCH_SIZE_2000),
            interpolation=cv2.INTER_NEAREST,
        )
        roll_1000_up = cv2.resize(
            roll_1000_crop,
            (PATCH_SIZE_2000, PATCH_SIZE_2000),
            interpolation=cv2.INTER_NEAREST,
        )
        cross_resolution_plate = panel_grid_2x2(
            [
                label_panel(plain_crop, "plain 2000 RAW"),
                label_panel(plain_1000_up, "plain 1000 RAW, 2x nearest display"),
                label_panel(roll_crop, "roll 2000 RAW"),
                label_panel(roll_1000_up, "roll 1000 RAW, 2x nearest display"),
            ]
        )
        cross_resolution_plate.save(plate_dir / f"{prefix}_cross_resolution.png")

        patch_records.append(
            {
                "patch_index": patch_index,
                "selection_basis": "500_ppi_ridge_SIFT_inlier_navigation_only",
                "navigation_score": patch["navigation_score"],
                "plain_center_500": patch["plain_center_500"].tolist(),
                "roll_center_500": patch["roll_center_500"].tolist(),
                "plain_box_2000_xyxy": list(plain_box),
                "roll_box_2000_xyxy": list(roll_box),
                "plain_box_1000_xyxy": list(plain_1000_box),
                "roll_box_1000_xyxy": list(roll_1000_box),
                "plain_to_roll_local_500": patch["plain_to_roll_local"].tolist(),
                "local_display_refinement": refinement_metrics,
                "raw_source_crops": {
                    "plain_2000": f"raw-crops/{prefix}_plain_2000_raw.png",
                    "roll_2000": f"raw-crops/{prefix}_roll_2000_raw.png",
                    "plain_1000": f"raw-crops/{prefix}_plain_1000_raw.png",
                    "roll_1000": f"raw-crops/{prefix}_roll_1000_raw.png",
                },
                "display_plates": {
                    "cross_impression": f"plates/{prefix}_cross_impression.png",
                    "cross_resolution": f"plates/{prefix}_cross_resolution.png",
                },
            }
        )

    metrics = {
        "sample_index": sample_index,
        "sample_key": sample_key,
        "subject_id": record["subject_id"],
        "anatomical_position": record["anatomical_position"],
        "finger_name": record["finger_name"],
        "source_dimensions_wh": {
            f"{impression}_{ppi}": [int(image.shape[1]), int(image.shape[0])]
            for (impression, ppi), image in images.items()
        },
        "navigation_alignment": {
            "method": "mutual_ratio_SIFT_on_500_ppi_then_USAC_MAGSAC_homography",
            "chosen_source_ppi_downsampled_to_500_for_navigation": alignment[
                "navigation_source_ppi"
            ],
            "candidate_alignments": alignment_candidates,
            "ratio_threshold_used": alignment["ratio"],
            "plain_keypoint_count": len(alignment["plain_keypoints"]),
            "roll_keypoint_count": len(alignment["roll_keypoints"]),
            "mutual_match_count": len(alignment["all_matches"]),
            "geometric_inlier_count": len(alignment["inliers"]),
            "geometrically_filtered_navigation_match_count": len(
                alignment["navigation_matches"]
            ),
            "median_inlier_error_500_px": alignment["median_inlier_error_500"],
            "p90_inlier_error_500_px": alignment["p90_inlier_error_500"],
            "plain_inlier_hull_fraction": alignment["plain_inlier_hull_fraction"],
            "roll_inlier_hull_fraction": alignment["roll_inlier_hull_fraction"],
            "warped_plain_mask_overlap_fraction": alignment[
                "warped_plain_mask_overlap_fraction"
            ],
            "navigation_quality_score": alignment["navigation_quality_score"],
            "plain_foreground_bbox_500_xywh": list(alignment["plain_bbox"]),
            "roll_foreground_bbox_500_xywh": list(alignment["roll_bbox"]),
            "plain_to_roll_homography_500": alignment["homography"].tolist(),
            "scientific_status": "NAVIGATION_ONLY_NOT_GROUND_TRUTH",
        },
        "patches": patch_records,
    }
    (sample_dir / "evidence_manifest.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in selection["selected"]:
        label = (
            f"S{record['sample_index']:02d} {record['subject_id']} "
            f"P{record['anatomical_position']:02d}"
        )
        try:
            result = process_record(record)
            results.append(result)
            nav = result["navigation_alignment"]
            print(
                f"{label}: inliers={nav['geometric_inlier_count']} "
                f"median_error={nav['median_inlier_error_500_px']:.2f}px "
                f"patches={len(result['patches'])}"
            )
        except Exception as error:  # keep the batch diagnostic and explicit
            failures.append(
                {
                    "sample_index": record["sample_index"],
                    "subject_id": record["subject_id"],
                    "anatomical_position": record["anatomical_position"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"{label}: FAILED: {type(error).__name__}: {error}")

    payload = {
        "experiment_id": selection["experiment_id"],
        "method_note": (
            "All alignments and patch scores are navigation aids only. Raw source "
            "crops retain original pixel values; aligned/contrast plates are display-only."
        ),
        "processed": results,
        "failures": failures,
    }
    output = ANALYSIS_ROOT / "navigation_alignment_metrics.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Processed={len(results)} Failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
