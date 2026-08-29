"""Ridge-only scale and registration primitives for Experiment 004 transfer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from fingerprint_new_method.experiment004 import (
    EXPERIMENT_ID,
    optimal_point_matching,
    repeatability,
    resolve_source_id,
    ridge_area_mask,
    sha256_file,
)


@dataclass(frozen=True)
class RidgePeriodEstimate:
    status: str
    period_px: float | None
    tile_periods: tuple[float, ...]
    tile_correlations: tuple[float, ...]
    candidate_tiles: int
    accepted_tiles: int
    mad_over_median: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "period_px": self.period_px,
            "tile_periods": list(self.tile_periods),
            "tile_correlations": list(self.tile_correlations),
            "candidate_tiles": self.candidate_tiles,
            "accepted_tiles": self.accepted_tiles,
            "mad_over_median": self.mad_over_median,
        }


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length < tile_size:
        return []
    values = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if values and values[-1] != final:
        values.append(final)
    return values


def _structure_tensor_orientation(tile: np.ndarray) -> tuple[float, float]:
    values = tile.astype(np.float32)
    smoothed = cv2.GaussianBlur(values, (0, 0), sigmaX=1.5, sigmaY=1.5)
    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float(np.sum(gradient_x * gradient_x))
    jyy = float(np.sum(gradient_y * gradient_y))
    jxy = float(np.sum(gradient_x * gradient_y))
    magnitude = math.hypot(jxx - jyy, 2.0 * jxy)
    coherence = magnitude / max(jxx + jyy, 1e-12)
    gradient_orientation = 0.5 * math.atan2(2.0 * jxy, jxx - jyy)
    ridge_orientation = gradient_orientation + math.pi / 2.0
    return ridge_orientation, coherence


def estimate_tile_ridge_period(tile: np.ndarray) -> tuple[float, float] | None:
    """Estimate ridge-to-ridge period in one frozen 256-pixel tile."""

    if tile.shape != (256, 256):
        raise ValueError(f"Expected 256x256 tile, got {tile.shape}")
    values = tile.astype(np.float32)
    if float(np.std(values)) < 8.0:
        return None
    orientation, coherence = _structure_tensor_orientation(values)
    if coherence < 0.20:
        return None
    rotation_degrees = 90.0 - math.degrees(orientation)
    matrix = cv2.getRotationMatrix2D((127.5, 127.5), rotation_degrees, 1.0)
    rotated = cv2.warpAffine(
        values,
        matrix,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    crop = rotated[32:224, 32:224]
    profile = np.mean(crop, axis=0).astype(np.float64)
    trend = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), sigmaX=24.0).reshape(-1)
    profile -= trend
    profile -= float(np.mean(profile))
    energy = float(np.dot(profile, profile))
    if energy <= 1e-9:
        return None
    autocorrelation = np.correlate(profile, profile, mode="full")[len(profile) - 1 :]
    normalizer = np.arange(len(profile), 0, -1, dtype=np.float64)
    autocorrelation = autocorrelation / normalizer
    autocorrelation /= max(float(autocorrelation[0]), 1e-12)
    candidates = [
        lag
        for lag in range(5, 65)
        if autocorrelation[lag] >= autocorrelation[lag - 1]
        and autocorrelation[lag] > autocorrelation[lag + 1]
    ]
    if not candidates:
        return None
    peak = max(float(autocorrelation[lag]) for lag in candidates)
    if peak < 0.15:
        return None
    selected = min(lag for lag in candidates if autocorrelation[lag] >= 0.90 * peak)
    return float(selected), float(autocorrelation[selected])


def estimate_ridge_period(gray: np.ndarray) -> RidgePeriodEstimate:
    from fingerprint_new_method.experiment004 import robust_normalize_uint8

    normalized = robust_normalize_uint8(gray)
    periods: list[float] = []
    correlations: list[float] = []
    y_starts = _tile_starts(normalized.shape[0], 256, 128)
    x_starts = _tile_starts(normalized.shape[1], 256, 128)
    candidate_tiles = len(y_starts) * len(x_starts)
    for y_value in y_starts:
        for x_value in x_starts:
            estimate = estimate_tile_ridge_period(normalized[y_value : y_value + 256, x_value : x_value + 256])
            if estimate is not None:
                periods.append(estimate[0])
                correlations.append(estimate[1])
    if len(periods) < 5:
        return RidgePeriodEstimate(
            "UNRELIABLE_TOO_FEW_TILES", None, tuple(periods), tuple(correlations), candidate_tiles, len(periods), None
        )
    median = float(np.median(periods))
    mad_ratio = float(np.median(np.abs(np.asarray(periods) - median)) / median)
    if mad_ratio > 0.25:
        return RidgePeriodEstimate(
            "UNRELIABLE_DISPERSION", None, tuple(periods), tuple(correlations), candidate_tiles, len(periods), mad_ratio
        )
    return RidgePeriodEstimate("OK", median, tuple(periods), tuple(correlations), candidate_tiles, len(periods), mad_ratio)


def frozen_training_ridge_period(
    train_records: Sequence[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    estimates: list[dict[str, Any]] = []
    accepted: list[float] = []
    for record in train_records:
        gray = cv2.imread(str(resolve_source_id(record["image_source_id"])), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not decode {record['image_source_id']}")
        estimate = estimate_ridge_period(gray)
        estimates.append({"canonical_image_id": record["canonical_image_id"], **estimate.as_dict()})
        if estimate.period_px is not None:
            accepted.append(estimate.period_px)
    if not accepted:
        raise RuntimeError("No reliable train ridge-period estimates")
    return float(np.median(accepted)), estimates


@dataclass(frozen=True)
class ScaledImage:
    status: str
    image: np.ndarray | None
    source_period_px: float | None
    target_period_px: float
    scale_factor: float | None
    estimate: RidgePeriodEstimate


#: Preregistered band of section 9; the primary analysis and Gate B use only this one.
FROZEN_SCALE_FACTOR_BAND = (0.20, 1.50)
#: Wide sanity guard of the frozen contingency amendment; exploratory sensitivity only.
CONFORMANT_SCALE_FACTOR_BAND = (0.20, 3.00)
SCALE_FACTOR_BANDS = {"frozen": FROZEN_SCALE_FACTOR_BAND, "conformant": CONFORMANT_SCALE_FACTOR_BAND}


def scale_factor_status(scale_factor: float | None, band: tuple[float, float]) -> str:
    """Classify one ridge-scale factor against an analysis band."""

    if scale_factor is None:
        return "PREPROCESSING_FAILURE"
    return "OK" if band[0] <= float(scale_factor) <= band[1] else "PREPROCESSING_FAILURE"


def normalize_ridge_scale(
    gray: np.ndarray,
    target_period_px: float,
    *,
    band: tuple[float, float] = FROZEN_SCALE_FACTOR_BAND,
) -> ScaledImage:
    estimate = estimate_ridge_period(gray)
    if estimate.period_px is None:
        return ScaledImage("PREPROCESSING_FAILURE", None, None, target_period_px, None, estimate)
    factor = float(target_period_px / estimate.period_px)
    if scale_factor_status(factor, band) != "OK":
        return ScaledImage("PREPROCESSING_FAILURE", None, estimate.period_px, target_period_px, factor, estimate)
    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=interpolation)
    return ScaledImage("OK", scaled, estimate.period_px, target_period_px, factor, estimate)


def _selection_source(sample: dict[str, Any], *, ppi: int, impression: str) -> dict[str, Any]:
    matches = [
        source for source in sample["sources"] if int(source["ppi"]) == ppi and source["impression"] == impression
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {ppi}-ppi {impression} source for sample {sample.get('sample_index')}, found {len(matches)}"
        )
    return matches[0]


def _nist_source_id(source_id: str) -> str:
    normalized = Path(source_id).as_posix()
    return normalized if normalized.startswith("NIST/") else f"NIST/{normalized}"


def negative_anatomical_position(position: int) -> int:
    if not 1 <= int(position) <= 10:
        raise ValueError(f"Anatomical position must be in 1..10, got {position}")
    return int(position) % 10 + 1


def build_sd300_pair_manifest(selection_manifest: dict[str, Any], *, ppis: Sequence[int] = (1000, 2000)) -> dict[str, Any]:
    selected = selection_manifest.get("selected")
    if not isinstance(selected, list) or len(selected) != 20:
        raise ValueError("Experiment 001 selection manifest must contain exactly 20 selected records")
    records: list[dict[str, Any]] = []
    roots = {1000: "sd300b", 2000: "sd300c"}
    for sample in selected:
        position = int(sample["anatomical_position"])
        negative_position = negative_anatomical_position(position)
        subject = str(sample["subject_id"])
        for ppi in ppis:
            if ppi not in roots:
                raise ValueError(f"Unsupported preregistered PPI: {ppi}")
            plain = _selection_source(sample, ppi=ppi, impression="plain")
            mated = _selection_source(sample, ppi=ppi, impression="roll")
            negative_relative = (
                f"NIST/{roots[ppi]}/images/{ppi}/png/roll/"
                f"{subject}_roll_{ppi}_{negative_position:02d}.png"
            )
            sources = {
                "plain_source_id": _nist_source_id(plain["source_id"]),
                "mated_roll_source_id": _nist_source_id(mated["source_id"]),
                "non_mated_roll_source_id": negative_relative,
            }
            existence = {key.replace("_source_id", "_exists"): resolve_source_id(value).is_file() for key, value in sources.items()}
            hashes = {
                key.replace("_source_id", "_sha256"): sha256_file(resolve_source_id(value)) if resolve_source_id(value).is_file() else None
                for key, value in sources.items()
            }
            records.append(
                {
                    "record_id": f"S{int(sample['sample_index']):02d}-{ppi}",
                    "sample_index": int(sample["sample_index"]),
                    "subject_id": subject,
                    "anatomical_position": position,
                    "non_mated_anatomical_position": negative_position,
                    "ppi": ppi,
                    **sources,
                    **existence,
                    **hashes,
                }
            )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "selection_source": "artifacts/experiment-001/selection/selection_manifest.json",
        "negative_rule": "i -> i+1; 10 -> 1, same subject",
        "records": records,
    }


def _mutual_ratio_matches(
    first_descriptors: np.ndarray,
    second_descriptors: np.ndarray,
    *,
    ratio: float = 0.78,
) -> list[cv2.DMatch]:
    if len(first_descriptors) < 2 or len(second_descriptors) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward_raw = matcher.knnMatch(first_descriptors, second_descriptors, k=2)
    backward_raw = matcher.knnMatch(second_descriptors, first_descriptors, k=2)
    forward = {pair[0].queryIdx: pair[0] for pair in forward_raw if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}
    backward = {
        pair[0].queryIdx: pair[0] for pair in backward_raw if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    matches = [
        match
        for query_index, match in forward.items()
        if match.trainIdx in backward and backward[match.trainIdx].trainIdx == query_index
    ]
    return sorted(matches, key=lambda item: (item.queryIdx, item.trainIdx, item.distance))


def _polygon_mask(shape: tuple[int, int], points: np.ndarray) -> tuple[np.ndarray, float]:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) < 3:
        return mask.astype(bool), 0.0
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    area = float(cv2.contourArea(hull))
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    return mask.astype(bool), area


def _perspective_coordinates(homography: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    if len(coordinates) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(np.asarray(coordinates, dtype=np.float32).reshape(1, -1, 2), homography)[0]


@dataclass
class Registration:
    status: str
    summary: dict[str, Any]
    homography: np.ndarray | None
    forward_flow: np.ndarray | None
    overlap_plain: np.ndarray
    overlap_roll: np.ndarray


def register_ridge_images(plain: np.ndarray, roll: np.ndarray) -> Registration:
    """Register already scale-normalized/preprocessed images without detector output."""

    if plain.ndim != 2 or roll.ndim != 2:
        raise ValueError("Registration inputs must be grayscale")
    plain_uint8 = np.clip(plain * 255.0 if plain.dtype.kind == "f" else plain, 0, 255).astype(np.uint8)
    roll_uint8 = np.clip(roll * 255.0 if roll.dtype.kind == "f" else roll, 0, 255).astype(np.uint8)
    plain_blurred = cv2.GaussianBlur(plain_uint8, (0, 0), sigmaX=2.0, sigmaY=2.0)
    roll_blurred = cv2.GaussianBlur(roll_uint8, (0, 0), sigmaX=2.0, sigmaY=2.0)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.6)
    first_keypoints, first_descriptors = sift.detectAndCompute(plain_blurred, None)
    second_keypoints, second_descriptors = sift.detectAndCompute(roll_blurred, None)
    empty_plain = np.zeros(plain.shape, dtype=bool)
    empty_roll = np.zeros(roll.shape, dtype=bool)
    if first_descriptors is None or second_descriptors is None:
        return Registration(
            "INVALID",
            {"reason": "NO_DESCRIPTORS", "mutual_matches": 0, "ransac_inliers": 0},
            None,
            None,
            empty_plain,
            empty_roll,
        )
    matches = _mutual_ratio_matches(first_descriptors, second_descriptors)
    if len(matches) < 4:
        return Registration(
            "INVALID",
            {"reason": "TOO_FEW_MATCHES", "mutual_matches": len(matches), "ransac_inliers": 0},
            None,
            None,
            empty_plain,
            empty_roll,
        )
    first_points = np.asarray([first_keypoints[item.queryIdx].pt for item in matches], dtype=np.float32)
    second_points = np.asarray([second_keypoints[item.trainIdx].pt for item in matches], dtype=np.float32)
    homography, inlier_mask_raw = cv2.findHomography(
        first_points,
        second_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=6.0,
        maxIters=10000,
        confidence=0.999,
    )
    if homography is None or inlier_mask_raw is None or not np.all(np.isfinite(homography)):
        return Registration(
            "INVALID",
            {"reason": "HOMOGRAPHY_FAILURE", "mutual_matches": len(matches), "ransac_inliers": 0},
            None,
            None,
            empty_plain,
            empty_roll,
        )
    inlier_mask = inlier_mask_raw.reshape(-1).astype(bool)
    inlier_first = first_points[inlier_mask]
    inlier_second = second_points[inlier_mask]
    projected = _perspective_coordinates(homography, inlier_first)
    errors = np.sqrt(np.sum((projected - inlier_second) ** 2, axis=1))
    plain_ridge_mask = ridge_area_mask(plain)
    roll_ridge_mask = ridge_area_mask(roll)
    first_hull_mask, first_hull_area = _polygon_mask(plain.shape, inlier_first)
    second_hull_mask, second_hull_area = _polygon_mask(roll.shape, inlier_second)
    first_coverage = first_hull_area / max(float(np.count_nonzero(plain_ridge_mask)), 1.0)
    second_coverage = second_hull_area / max(float(np.count_nonzero(roll_ridge_mask)), 1.0)
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / len(matches)
    median_error = float(np.median(errors)) if len(errors) else None
    p90_error = float(np.percentile(errors, 90)) if len(errors) else None
    valid = (
        len(matches) >= 20
        and inlier_count >= 12
        and inlier_ratio >= 0.25
        and median_error is not None
        and median_error <= 4.0
        and p90_error is not None
        and p90_error <= 8.0
        and first_coverage >= 0.08
        and second_coverage >= 0.08
    )
    ambiguous = len(matches) >= 8 and inlier_count >= 6
    status = "VALID" if valid else ("AMBIGUOUS" if ambiguous else "INVALID")
    summary: dict[str, Any] = {
        "reason": "ALL_VALIDITY_THRESHOLDS_MET" if valid else "VALIDITY_THRESHOLDS_NOT_MET",
        "plain_keypoints": len(first_keypoints),
        "roll_keypoints": len(second_keypoints),
        "mutual_matches": len(matches),
        "ransac_inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "median_reprojection_error": median_error,
        "p90_reprojection_error": p90_error,
        "plain_hull_coverage": first_coverage,
        "roll_hull_coverage": second_coverage,
    }
    if status != "VALID":
        return Registration(status, summary, homography, None, empty_plain, empty_roll)

    height, width = plain.shape
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    plain_grid = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    mapped_grid = _perspective_coordinates(homography, plain_grid).reshape(height, width, 2)
    aligned_roll = cv2.remap(
        roll_blurred,
        mapped_grid[:, :, 0],
        mapped_grid[:, :, 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    forward_flow = cv2.calcOpticalFlowFarneback(
        plain_blurred,
        aligned_roll,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=31,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    backward_flow = cv2.calcOpticalFlowFarneback(
        aligned_roll,
        plain_blurred,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=31,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    shifted_x = grid_x + forward_flow[:, :, 0]
    shifted_y = grid_y + forward_flow[:, :, 1]
    backward_x = cv2.remap(backward_flow[:, :, 0], shifted_x, shifted_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    backward_y = cv2.remap(backward_flow[:, :, 1], shifted_x, shifted_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    consistency = np.hypot(forward_flow[:, :, 0] + backward_x, forward_flow[:, :, 1] + backward_y) <= 2.0
    shifted_plain = np.column_stack([shifted_x.reshape(-1), shifted_y.reshape(-1)])
    refined_roll = _perspective_coordinates(homography, shifted_plain).reshape(height, width, 2)
    inside_roll = (
        (refined_roll[:, :, 0] >= 0)
        & (refined_roll[:, :, 0] <= roll.shape[1] - 1)
        & (refined_roll[:, :, 1] >= 0)
        & (refined_roll[:, :, 1] <= roll.shape[0] - 1)
    )
    mapped_roll_ridge = cv2.remap(
        roll_ridge_mask.astype(np.uint8),
        refined_roll[:, :, 0],
        refined_roll[:, :, 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    mapped_second_hull = cv2.remap(
        second_hull_mask.astype(np.uint8),
        refined_roll[:, :, 0],
        refined_roll[:, :, 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    overlap_plain = plain_ridge_mask & first_hull_mask & mapped_roll_ridge & mapped_second_hull & inside_roll & consistency
    overlap_roll = np.zeros(roll.shape, dtype=np.uint8)
    mapped_overlap = refined_roll[overlap_plain]
    if len(mapped_overlap):
        rounded = np.rint(mapped_overlap).astype(np.int64)
        valid_coordinates = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < roll.shape[1])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < roll.shape[0])
        )
        rounded = rounded[valid_coordinates]
        overlap_roll[rounded[:, 1], rounded[:, 0]] = 1
        overlap_roll = cv2.dilate(overlap_roll, np.ones((3, 3), dtype=np.uint8))
        overlap_roll &= roll_ridge_mask.astype(np.uint8)
        overlap_roll &= second_hull_mask.astype(np.uint8)
    summary["overlap_plain_pixels"] = int(np.count_nonzero(overlap_plain))
    summary["overlap_roll_pixels"] = int(np.count_nonzero(overlap_roll))
    if summary["overlap_plain_pixels"] == 0 or summary["overlap_roll_pixels"] == 0:
        summary["reason"] = "EMPTY_MUTUAL_OVERLAP"
        return Registration("INVALID", summary, homography, forward_flow, empty_plain, empty_roll)
    return Registration("VALID", summary, homography, forward_flow, overlap_plain, overlap_roll.astype(bool))


def map_plain_to_roll(registration: Registration, points: np.ndarray) -> np.ndarray:
    if registration.status != "VALID" or registration.homography is None or registration.forward_flow is None:
        raise ValueError("A VALID registration is required")
    coordinates = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(coordinates) == 0:
        return np.empty((0, 2), dtype=np.float32)
    map_x = coordinates[:, 0].reshape(-1, 1)
    map_y = coordinates[:, 1].reshape(-1, 1)
    flow_x = cv2.remap(
        registration.forward_flow[:, :, 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    ).reshape(-1)
    flow_y = cv2.remap(
        registration.forward_flow[:, :, 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    ).reshape(-1)
    shifted = coordinates + np.column_stack([flow_x, flow_y])
    return _perspective_coordinates(registration.homography, shifted)


def _points_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(coordinates) == 0:
        return np.empty((0,), dtype=bool)
    rounded = np.rint(coordinates).astype(np.int64)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < mask.shape[0])
    )
    result = np.zeros(len(coordinates), dtype=bool)
    selected = rounded[valid]
    result[np.flatnonzero(valid)] = mask[selected[:, 1], selected[:, 0]]
    return result


def score_registered_detections(
    registration: Registration,
    plain_detections: np.ndarray,
    roll_detections: np.ndarray,
    *,
    tolerance: float = 4.0,
) -> dict[str, Any]:
    if registration.status != "VALID":
        return {
            "status": f"REGISTRATION_{registration.status}",
            "repeatability": None,
            "matched": None,
            "plain_in_overlap": None,
            "roll_in_overlap": None,
        }
    plain = np.asarray(plain_detections, dtype=np.float32).reshape(-1, 2)
    roll = np.asarray(roll_detections, dtype=np.float32).reshape(-1, 2)
    plain_selected = plain[_points_in_mask(plain, registration.overlap_plain)]
    roll_selected = roll[_points_in_mask(roll, registration.overlap_roll)]
    mapped_plain = map_plain_to_roll(registration, plain_selected)
    matches = optimal_point_matching(mapped_plain, roll_selected, tolerance)
    metric = repeatability(len(matches), len(plain_selected), len(roll_selected))
    distances = [item[2] for item in matches]
    return {
        "status": metric["status"],
        "repeatability": metric["repeatability"],
        "matched": len(matches),
        "plain_in_overlap": len(plain_selected),
        "roll_in_overlap": len(roll_selected),
        "mean_match_distance": float(np.mean(distances)) if distances else None,
    }


def registration_to_npz(path: Path, registration: Registration) -> None:
    if registration.homography is None:
        homography = np.empty((0, 0), dtype=np.float64)
    else:
        homography = registration.homography
    if registration.forward_flow is None:
        flow = np.empty((0, 0, 2), dtype=np.float32)
    else:
        flow = registration.forward_flow
    metadata = json.dumps({"status": registration.status, "summary": registration.summary}, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(metadata),
        homography=homography,
        forward_flow=flow,
        overlap_plain=registration.overlap_plain.astype(np.uint8),
        overlap_roll=registration.overlap_roll.astype(np.uint8),
    )


def registration_from_npz(path: Path) -> Registration:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        homography = archive["homography"]
        flow = archive["forward_flow"]
        return Registration(
            status=metadata["status"],
            summary=metadata["summary"],
            homography=homography if homography.size else None,
            forward_flow=flow if flow.size else None,
            overlap_plain=archive["overlap_plain"].astype(bool),
            overlap_roll=archive["overlap_roll"].astype(bool),
        )
