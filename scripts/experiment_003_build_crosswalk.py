"""Build the preregistered Experiment 003 L3-SF identity crosswalk.

The script implements two pore-independent channels: RootSIFT keypoints and
dense ridge-orientation patches.  Large impression-level matrices remain in
the ignored local-large directory; compact crosswalk evidence is tracked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from fingerprint_new_method.experiment003 import (
    maximum_weight_assignment,
    parse_annotated_filename,
    parse_final_filename,
    write_csv,
    write_json,
)
from fingerprint_new_method.paths import PROJECT_ROOT, dataset_path

EXPERIMENT_ID = "003-l3sf-annotated-final-crosswalk"
DATA_ROOT = dataset_path("L3_SF_V2", "L3SF_V2")
FINAL_ROOT = DATA_ROOT / "L3-SF"
ANNOTATED_ROOT = DATA_ROOT / "Pore ground truth" / "Fingerprint Images"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "experiment-003"
LOCAL_ROOT = OUTPUT_ROOT / "local-large"
RUNS = tuple(f"R{index}" for index in range(1, 6))
METRIC_NAMES = ("score", "correspondences", "inliers", "inlier_ratio", "residual", "coverage_query", "coverage_final", "mean_weight")


@dataclass(slots=True)
class Features:
    descriptors: np.ndarray
    coordinates: np.ndarray
    coordinate_area: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not decode {path}")
    low, high = np.percentile(gray, (1.0, 99.0))
    if high <= low:
        normalized = np.zeros_like(gray)
    else:
        normalized = np.clip((gray.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)


def rootsift_features(gray: np.ndarray, sift: cv2.SIFT) -> Features:
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        return Features(np.empty((0, 128), np.float32), np.empty((0, 2), np.float32), float(gray.size))
    ordering = sorted(
        range(len(keypoints)),
        key=lambda index: (
            -keypoints[index].response,
            keypoints[index].pt[1],
            keypoints[index].pt[0],
            keypoints[index].size,
            keypoints[index].angle,
        ),
    )
    descriptors = descriptors[np.asarray(ordering)]
    descriptors /= np.maximum(np.sum(np.abs(descriptors), axis=1, keepdims=True), 1e-12)
    descriptors = np.sqrt(descriptors).astype(np.float32)
    coordinates = np.asarray([keypoints[index].pt for index in ordering], dtype=np.float32)
    return Features(np.ascontiguousarray(descriptors), coordinates, float(gray.shape[0] * gray.shape[1]))


def orientation_features(gray: np.ndarray) -> Features:
    smoothed = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
    gradient_x = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1)
    height_blocks = gray.shape[0] // 8
    width_blocks = gray.shape[1] // 8

    def block_sum(values: np.ndarray) -> np.ndarray:
        cropped = values[: height_blocks * 8, : width_blocks * 8]
        return cropped.reshape(height_blocks, 8, width_blocks, 8).sum(axis=(1, 3))

    jxx = block_sum(gradient_x * gradient_x)
    jyy = block_sum(gradient_y * gradient_y)
    jxy = block_sum(gradient_x * gradient_y)
    magnitude = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2)
    coherence = magnitude / np.maximum(jxx + jyy, 1e-12)
    gradient_orientation = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    ridge_orientation = gradient_orientation + np.float32(np.pi / 2.0)
    field = np.stack(
        [np.cos(2.0 * ridge_orientation), np.sin(2.0 * ridge_orientation), coherence], axis=2
    ).astype(np.float32)
    padded = np.pad(field, ((2, 2), (2, 2), (0, 0)), mode="reflect")
    candidate_y, candidate_x = np.nonzero(coherence >= 0.15)
    if len(candidate_x) == 0:
        return Features(np.empty((0, 75), np.float32), np.empty((0, 2), np.float32), float(height_blocks * width_blocks))
    ordering = np.lexsort((candidate_x, candidate_y, -coherence[candidate_y, candidate_x]))[:512]
    candidate_y = candidate_y[ordering]
    candidate_x = candidate_x[ordering]
    descriptors = np.empty((len(candidate_x), 75), dtype=np.float32)
    for index, (y_value, x_value) in enumerate(zip(candidate_y, candidate_x, strict=True)):
        descriptor = padded[y_value : y_value + 5, x_value : x_value + 5].reshape(-1).copy()
        descriptor -= float(np.mean(descriptor))
        descriptor /= max(float(np.sqrt(np.sum(descriptor * descriptor))), 1e-12)
        descriptors[index] = descriptor
    coordinates = np.column_stack([candidate_x.astype(np.float32) + 0.5, candidate_y.astype(np.float32) + 0.5])
    return Features(np.ascontiguousarray(descriptors), coordinates, float(height_blocks * width_blocks))


def concatenate_final_features(features: list[Features]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    descriptor_arrays = [feature.descriptors for feature in features if len(feature.descriptors)]
    if not descriptor_arrays:
        raise RuntimeError("No final descriptors were extracted")
    descriptors = np.ascontiguousarray(np.concatenate(descriptor_arrays, axis=0), dtype=np.float32)
    coordinates = np.concatenate([feature.coordinates for feature in features if len(feature.descriptors)], axis=0)
    owners = np.concatenate(
        [np.full(len(feature.descriptors), index, dtype=np.int32) for index, feature in enumerate(features) if len(feature.descriptors)]
    )
    areas = np.asarray([feature.coordinate_area for feature in features], dtype=np.float32)
    return descriptors, coordinates, owners, areas


def convex_hull_coverage(points: np.ndarray, area: float) -> float:
    if len(points) < 3 or area <= 0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return max(0.0, min(1.0, float(cv2.contourArea(hull)) / area))


def affine_residuals(matrix: np.ndarray, query_points: np.ndarray, final_points: np.ndarray) -> np.ndarray:
    predicted_x = matrix[0, 0] * query_points[:, 0] + matrix[0, 1] * query_points[:, 1] + matrix[0, 2]
    predicted_y = matrix[1, 0] * query_points[:, 0] + matrix[1, 1] * query_points[:, 1] + matrix[1, 2]
    return np.sqrt((predicted_x - final_points[:, 0]) ** 2 + (predicted_y - final_points[:, 1]) ** 2)


def score_correspondences(
    query_points: np.ndarray,
    final_points: np.ndarray,
    weights: np.ndarray,
    *,
    query_area: float,
    final_area: float,
    ransac_threshold: float,
    residual_penalty: float,
) -> tuple[float, int, int, float, float, float, float, float]:
    count = len(query_points)
    mean_weight = float(np.mean(weights)) if count else 0.0
    if count < 4:
        return (0.0, count, 0, 0.0, math.nan, 0.0, 0.0, mean_weight)
    matrix, mask = cv2.estimateAffinePartial2D(
        query_points,
        final_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.995,
        refineIters=10,
    )
    if matrix is None or mask is None:
        return (0.0, count, 0, 0.0, math.nan, 0.0, 0.0, mean_weight)
    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inlier_mask))
    if inlier_count == 0:
        return (0.0, count, 0, 0.0, math.nan, 0.0, 0.0, mean_weight)
    inlier_ratio = inlier_count / count
    residual = float(np.median(affine_residuals(matrix, query_points[inlier_mask], final_points[inlier_mask])))
    coverage_query = convex_hull_coverage(query_points[inlier_mask], query_area)
    coverage_final = convex_hull_coverage(final_points[inlier_mask], final_area)
    score = (
        inlier_count
        * inlier_ratio
        * math.sqrt(coverage_query * coverage_final)
        * math.exp(-residual / residual_penalty)
    )
    return (score, count, inlier_count, inlier_ratio, residual, coverage_query, coverage_final, mean_weight)


def retrieve_and_score(
    query: Features,
    final_features: list[Features],
    final_coordinates: np.ndarray,
    final_owners: np.ndarray,
    final_areas: np.ndarray,
    index: cv2.flann_Index,
    *,
    ransac_threshold: float,
    residual_penalty: float,
) -> np.ndarray:
    output = np.zeros((len(final_features), len(METRIC_NAMES)), dtype=np.float32)
    output[:, 4] = np.nan
    if len(query.descriptors) == 0:
        return output
    neighbor_count = min(128, len(final_coordinates))
    indices, distances = index.knnSearch(query.descriptors, neighbor_count, params={"checks": 256})
    buckets: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for query_index in range(len(query.descriptors)):
        scale = max(float(distances[query_index, neighbor_count - 1]), 1e-9)
        seen_owners: set[int] = set()
        for neighbor_position in range(neighbor_count):
            descriptor_index = int(indices[query_index, neighbor_position])
            owner = int(final_owners[descriptor_index])
            if owner in seen_owners:
                continue
            seen_owners.add(owner)
            distance = float(distances[query_index, neighbor_position])
            weight = math.exp(-0.5 * (distance / scale) ** 2)
            buckets[owner].append((query_index, descriptor_index, weight))
    for owner, matches in buckets.items():
        query_indices = np.fromiter((match[0] for match in matches), dtype=np.int32)
        final_indices = np.fromiter((match[1] for match in matches), dtype=np.int32)
        weights = np.fromiter((match[2] for match in matches), dtype=np.float32)
        output[owner] = score_correspondences(
            query.coordinates[query_indices],
            final_coordinates[final_indices],
            weights,
            query_area=query.coordinate_area,
            final_area=float(final_areas[owner]),
            ransac_threshold=ransac_threshold,
            residual_penalty=residual_penalty,
        )
    return output


def list_run_images(run: str) -> tuple[list[tuple[Path, Any]], list[tuple[Path, Any]]]:
    final_images = [(path, parse_final_filename(path.name)) for path in (FINAL_ROOT / run).glob("*.png")]
    final_images.sort(key=lambda item: (item[1].identity, item[1].capture_group, item[1].instance))
    annotated_images = [(path, parse_annotated_filename(path.name)) for path in (ANNOTATED_ROOT / run).glob("*.jpg")]
    annotated_images.sort(key=lambda item: (item[0].stem, item[1].pattern, item[1].local_index))
    if len(final_images) != 1480 or len(annotated_images) != 148:
        raise RuntimeError(f"Unexpected cardinality for {run}: {len(final_images)} final, {len(annotated_images)} annotated")
    return final_images, annotated_images


def process_channel(
    run: str,
    final_images: list[tuple[Path, Any]],
    annotated_images: list[tuple[Path, Any]],
    *,
    channel: str,
) -> dict[str, np.ndarray]:
    if channel == "k":
        sift = cv2.SIFT_create(nfeatures=384, contrastThreshold=0.02, edgeThreshold=10, sigma=1.6)

        def extractor(gray: np.ndarray) -> Features:
            return rootsift_features(gray, sift)

        threshold = 4.0
        penalty = 4.0
    elif channel == "o":
        extractor = orientation_features
        threshold = 1.5
        penalty = 1.5
    else:
        raise ValueError(channel)

    started = time.perf_counter()
    final_features: list[Features] = []
    for index_value, (path, _) in enumerate(final_images, start=1):
        final_features.append(extractor(preprocess(path)))
        if index_value % 250 == 0:
            print(f"{run} {channel}: extracted final features {index_value}/{len(final_images)}", flush=True)
    descriptors, coordinates, owners, areas = concatenate_final_features(final_features)
    cv2.setRNGSeed(3003)
    flann_index = cv2.flann_Index(descriptors, {"algorithm": 1, "trees": 8})
    print(
        f"{run} {channel}: index has {len(descriptors)} descriptors; extraction {time.perf_counter() - started:.1f}s",
        flush=True,
    )

    matrices = {name: np.zeros((len(annotated_images), len(final_images)), dtype=np.float32) for name in METRIC_NAMES}
    matrices["residual"][:] = np.nan
    for annotated_index, (path, _) in enumerate(annotated_images):
        cv2.setRNGSeed(3003 + annotated_index)
        query = extractor(preprocess(path))
        scores = retrieve_and_score(
            query,
            final_features,
            coordinates,
            owners,
            areas,
            flann_index,
            ransac_threshold=threshold,
            residual_penalty=penalty,
        )
        for metric_index, name in enumerate(METRIC_NAMES):
            matrices[name][annotated_index] = scores[:, metric_index]
        if (annotated_index + 1) % 10 == 0 or annotated_index + 1 == len(annotated_images):
            print(
                f"{run} {channel}: scored annotated {annotated_index + 1}/{len(annotated_images)}; "
                f"elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return matrices


def process_run(run: str, *, overwrite: bool) -> Path:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = LOCAL_ROOT / f"{run.lower()}_impression_scores.npz"
    if output_path.exists() and not overwrite:
        print(f"{run}: reusing {output_path}", flush=True)
        return output_path
    final_images, annotated_images = list_run_images(run)
    channel_matrices: dict[str, np.ndarray] = {}
    for channel in ("k", "o"):
        matrices = process_channel(run, final_images, annotated_images, channel=channel)
        channel_matrices.update({f"{channel}_{name}": values for name, values in matrices.items()})
    metadata: dict[str, np.ndarray] = {
        "annotated_ids": np.asarray([f"{run}/{path.stem}" for path, _ in annotated_images]),
        "annotated_patterns": np.asarray([parsed.pattern for _, parsed in annotated_images]),
        "final_sample_ids": np.asarray([f"{run}/{path.stem}" for path, _ in final_images]),
        "final_identities": np.asarray([parsed.identity for _, parsed in final_images], dtype=np.int16),
        "capture_groups": np.asarray([parsed.capture_group for _, parsed in final_images], dtype=np.int8),
        "instances": np.asarray([parsed.instance for _, parsed in final_images], dtype=np.int8),
    }
    np.savez_compressed(output_path, **metadata, **channel_matrices)
    print(f"{run}: wrote {output_path} ({output_path.stat().st_size} bytes)", flush=True)
    return output_path


def top_three_mean(scores: np.ndarray) -> np.ndarray:
    if scores.shape[-1] < 3:
        raise ValueError("At least three impression scores are required")
    partitioned = np.partition(scores, scores.shape[-1] - 3, axis=-1)
    return np.mean(partitioned[..., -3:], axis=-1)


def winner_and_runner_up(scores: np.ndarray) -> tuple[int, float, int, float]:
    identities = np.arange(1, len(scores) + 1)
    ordering = np.lexsort((identities, -scores))
    first, second = int(ordering[0]), int(ordering[1])
    return first + 1, float(scores[first]), second + 1, float(scores[second])


def robust_separation(scores: np.ndarray, top_score: float, second_score: float) -> float:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    return (top_score - second_score) / max(1.4826 * mad, 1e-9)


def robust_z_rows(scores: np.ndarray) -> np.ndarray:
    medians = np.median(scores, axis=1, keepdims=True)
    mads = np.median(np.abs(scores - medians), axis=1, keepdims=True)
    return (scores - medians) / np.maximum(1.4826 * mads, 1e-9)


def leave_one_winner_stable(impression_scores: np.ndarray, winner_index: int, identity_scores: np.ndarray) -> bool:
    for position in range(10):
        retained = np.delete(impression_scores[winner_index], position)
        winner_score = float(top_three_mean(retained[None, :])[0])
        modified = identity_scores.copy()
        modified[winner_index] = winner_score
        if winner_and_runner_up(modified)[0] != winner_index + 1:
            return False
    return True


def support_count(arrays: dict[str, np.ndarray], row: int, identity_index: int, channel: str) -> int:
    selection = slice(identity_index * 10, identity_index * 10 + 10)
    inlier_minimum = 6 if channel == "k" else 10
    coverage_minimum = 0.01 if channel == "k" else 0.03
    passes = (
        (arrays[f"{channel}_inliers"][row, selection] >= inlier_minimum)
        & (arrays[f"{channel}_inlier_ratio"][row, selection] >= 0.15)
        & (arrays[f"{channel}_coverage_query"][row, selection] >= coverage_minimum)
        & (arrays[f"{channel}_coverage_final"][row, selection] >= coverage_minimum)
    )
    return int(np.count_nonzero(passes))


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def summarize_counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def finalize(run_paths: list[Path]) -> None:
    if len(run_paths) != 5 or any(not path.exists() for path in run_paths):
        raise RuntimeError("All five run score files are required for finalization")
    manifest_files: list[dict[str, Any]] = []
    crosswalk_rows: list[dict[str, Any]] = []
    run_summaries: dict[str, Any] = {}

    for run, path in zip(RUNS, run_paths, strict=True):
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        annotated_ids = arrays["annotated_ids"].tolist()
        patterns = arrays["annotated_patterns"].tolist()
        final_identities = arrays["final_identities"]
        expected_identities = np.repeat(np.arange(1, 149), 10)
        if not np.array_equal(final_identities, expected_identities):
            raise RuntimeError(f"Unexpected final identity ordering in {path}")

        channel_group_scores: dict[str, np.ndarray] = {}
        channel_impression_scores: dict[str, np.ndarray] = {}
        for channel in ("k", "o"):
            impression_scores = arrays[f"{channel}_score"].reshape(148, 148, 10)
            channel_impression_scores[channel] = impression_scores
            channel_group_scores[channel] = top_three_mean(impression_scores)
        combined = (robust_z_rows(channel_group_scores["k"]) + robust_z_rows(channel_group_scores["o"])) / 2.0
        assignment = maximum_weight_assignment(combined.tolist())

        provisional: list[bool] = []
        local_records: list[dict[str, Any]] = []
        for row_index, annotated_id in enumerate(annotated_ids):
            record: dict[str, Any] = {
                "annotated_512_id": annotated_id,
                "run": run,
                "pattern": patterns[row_index],
            }
            winners: dict[str, int] = {}
            all_channel_pass = True
            for channel in ("k", "o"):
                scores = channel_group_scores[channel][row_index]
                winner, winner_score, runner, runner_score = winner_and_runner_up(scores)
                winners[channel] = winner
                separation = robust_separation(scores, winner_score, runner_score)
                ratio_pass = winner_score > 0 and (runner_score == 0 or winner_score >= 1.25 * runner_score)
                margin_pass = separation >= 2.5 and ratio_pass
                count = support_count(arrays, row_index, winner - 1, channel)
                support_pass = count >= (2 if channel == "k" else 3)
                loo_stable = leave_one_winner_stable(
                    channel_impression_scores[channel][row_index], winner - 1, scores
                )
                dropped_winners = []
                for position in range(10):
                    retained = np.delete(channel_impression_scores[channel][row_index], position, axis=1)
                    dropped_scores = top_three_mean(retained)
                    dropped_winners.append(winner_and_runner_up(dropped_scores)[0])
                drop_change_count = sum(item != winner for item in dropped_winners)
                record.update(
                    {
                        f"{channel}_top1_identity": winner,
                        f"{channel}_top1_score": winner_score,
                        f"{channel}_top2_identity": runner,
                        f"{channel}_top2_score": runner_score,
                        f"{channel}_robust_separation": separation,
                        f"{channel}_support_impression_count": count,
                        f"{channel}_support_pass": bool_text(support_pass),
                        f"{channel}_margin_pass": bool_text(margin_pass),
                        f"{channel}_winner_leave_one_out_stable": bool_text(loo_stable),
                        f"{channel}_global_position_drop_change_count": drop_change_count,
                    }
                )
                all_channel_pass = all_channel_pass and support_pass and margin_pass and loo_stable
            channels_agree = winners["k"] == winners["o"]
            assigned_identity = assignment[row_index] + 1
            assignment_agrees = channels_agree and assigned_identity == winners["k"]
            record.update(
                {
                    "channels_agree": bool_text(channels_agree),
                    "global_assignment_identity": assigned_identity,
                    "assignment_agrees_with_local": bool_text(assignment_agrees),
                }
            )
            provisional.append(all_channel_pass and channels_agree and assignment_agrees)
            local_records.append(record)

        winner_counts = Counter(
            int(record["k_top1_identity"])
            for record, qualifies in zip(local_records, provisional, strict=True)
            if qualifies
        )
        for record, qualifies in zip(local_records, provisional, strict=True):
            winner = int(record["k_top1_identity"])
            collision = qualifies and winner_counts[winner] > 1
            strong = qualifies and not collision
            record["one_to_one_collision"] = bool_text(collision)
            record["status"] = "STRONG" if strong else "AMBIGUOUS"
            record["frozen_final_identity"] = winner if strong else ""
            crosswalk_rows.append(record)

        run_records = crosswalk_rows[-148:]
        run_summaries[run] = {
            "annotated_count": 148,
            "strong_count": sum(record["status"] == "STRONG" for record in run_records),
            "ambiguous_count": sum(record["status"] == "AMBIGUOUS" for record in run_records),
            "channel_agreement_count": sum(record["channels_agree"] == "true" for record in run_records),
            "assignment_agreement_count": sum(record["assignment_agrees_with_local"] == "true" for record in run_records),
            "collision_count": sum(record["one_to_one_collision"] == "true" for record in run_records),
            "status_by_pattern": {
                pattern: summarize_counter(record["status"] for record in run_records if record["pattern"] == pattern)
                for pattern in sorted(set(patterns))
            },
        }
        manifest_files.append(
            {
                "name": path.relative_to(PROJECT_ROOT).as_posix(),
                "rows_per_channel": 148 * 1480,
                "shape_per_metric": [148, 1480],
                "metric_names": list(METRIC_NAMES),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    fieldnames = [
        "annotated_512_id",
        "run",
        "pattern",
        "k_top1_identity",
        "k_top1_score",
        "k_top2_identity",
        "k_top2_score",
        "k_robust_separation",
        "k_support_impression_count",
        "k_support_pass",
        "k_margin_pass",
        "k_winner_leave_one_out_stable",
        "k_global_position_drop_change_count",
        "o_top1_identity",
        "o_top1_score",
        "o_top2_identity",
        "o_top2_score",
        "o_robust_separation",
        "o_support_impression_count",
        "o_support_pass",
        "o_margin_pass",
        "o_winner_leave_one_out_stable",
        "o_global_position_drop_change_count",
        "channels_agree",
        "global_assignment_identity",
        "assignment_agrees_with_local",
        "one_to_one_collision",
        "status",
        "frozen_final_identity",
    ]
    crosswalk_path = OUTPUT_ROOT / "crosswalk.csv"
    write_csv(crosswalk_path, crosswalk_rows, fieldnames)
    status_counts = summarize_counter(record["status"] for record in crosswalk_rows)
    strong_count = status_counts.get("STRONG", 0)
    channel_agreement_count = sum(record["channels_agree"] == "true" for record in crosswalk_rows)
    automatic_threshold_pass = strong_count >= 703
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "crosswalk_revision": 1,
        "candidate_scope": "same_run_only_148_identities_10_impressions_each",
        "pore_annotations_used": False,
        "annotated_count": len(crosswalk_rows),
        "status_counts": status_counts,
        "strong_fraction": strong_count / len(crosswalk_rows),
        "channel_agreement_count": channel_agreement_count,
        "channel_agreement_fraction": channel_agreement_count / len(crosswalk_rows),
        "assignment_agreement_count": sum(
            record["assignment_agrees_with_local"] == "true" for record in crosswalk_rows
        ),
        "collision_count": sum(record["one_to_one_collision"] == "true" for record in crosswalk_rows),
        "automatic_95pct_threshold_pass": automatic_threshold_pass,
        "gate_1_status": "PENDING_FROZEN_VISUAL_REVIEW" if automatic_threshold_pass else "FAIL_AUTOMATIC_EVIDENCE",
        "per_run": run_summaries,
        "crosswalk_sha256": sha256_file(crosswalk_path),
    }
    write_json(OUTPUT_ROOT / "crosswalk_summary.json", summary)
    write_json(
        OUTPUT_ROOT / "score_matrices.manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "tracked_in_git": False,
            "files": manifest_files,
            "regeneration_command": "& .\\.conda-env\\python.exe .\\scripts\\experiment_003_build_crosswalk.py --overwrite",
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_paths = [process_run(run, overwrite=args.overwrite) for run in args.runs]
    if not args.no_finalize:
        all_paths = [LOCAL_ROOT / f"{run.lower()}_impression_scores.npz" for run in RUNS]
        finalize(all_paths if set(args.runs) != set(RUNS) else selected_paths)


if __name__ == "__main__":
    main()
