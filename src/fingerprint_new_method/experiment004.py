"""Dataset-independent primitives for Experiment 004.

The module deliberately has no PyTorch dependency.  Dataset access is limited to
the annotated L3-SF branch and is resolved through :mod:`fingerprint_new_method.paths`.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from fingerprint_new_method.paths import PROJECT_ROOT, dataset_path, datasets_root

EXPERIMENT_ID = "004-pore-localization-and-sd300-transfer"
SPLIT_SEED = "exp004-l3sf-grouped-stratified-split-v1"
TRAINING_SEEDS = (40401, 40402, 40403)
BOOTSTRAP_SEED = 4049001
RUNS = tuple(f"R{index}" for index in range(1, 6))
PATTERN_QUOTAS: dict[str, dict[str, int]] = {
    "right_loop": {"train": 44, "validation": 15, "test": 15},
    "whorl": {"train": 36, "validation": 12, "test": 12},
    "left_loop": {"train": 3, "validation": 1, "test": 1},
    "plain_arch": {"train": 4, "validation": 1, "test": 1},
    "tented_arch": {"train": 1, "validation": 1, "test": 1},
}
PATTERNS = tuple(PATTERN_QUOTAS)
ANNOTATED_IMAGES_RELATIVE = (
    "L3_SF_V2",
    "L3SF_V2",
    "Pore ground truth",
    "Fingerprint Images",
)
ANNOTATIONS_RELATIVE = (
    "L3_SF_V2",
    "L3SF_V2",
    "Pore ground truth",
    "Ground truth",
)
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "experiment-004"
LOCAL_ROOT = ARTIFACT_ROOT / "local-large"
FILENAME_RE = re.compile(
    r"^(?P<local_index>[1-9][0-9]*)_(?P<pattern>right_loop|whorl|left_loop|plain_arch|tented_arch)$"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def write_json(path: Path, value: Any) -> None:
    """Atomically write pretty, strict JSON below the project workspace."""

    resolved = path.resolve()
    if PROJECT_ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError(f"Refusing to write outside project root: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    resolved = path.resolve()
    if PROJECT_ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError(f"Refusing to write outside project root: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_annotated_stem(stem: str) -> tuple[int, str]:
    match = FILENAME_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unexpected annotated filename stem: {stem}")
    return int(match.group("local_index")), match.group("pattern")


def split_rank(pattern: str, local_index: int, seed: str = SPLIT_SEED) -> str:
    return sha256_bytes(f"{seed}|{pattern}|{local_index}".encode())


def assign_grouped_split(
    groups: Iterable[tuple[str, int]],
    *,
    seed: str = SPLIT_SEED,
    quotas: dict[str, dict[str, int]] = PATTERN_QUOTAS,
) -> dict[tuple[str, int], dict[str, str]]:
    """Assign conservative pattern/index groups using the preregistered hash order."""

    grouped: dict[str, set[int]] = defaultdict(set)
    for pattern, local_index in groups:
        if pattern not in quotas:
            raise ValueError(f"Unexpected pattern: {pattern}")
        grouped[pattern].add(int(local_index))
    if set(grouped) != set(quotas):
        raise ValueError(f"Pattern mismatch: observed={sorted(grouped)}, expected={sorted(quotas)}")

    assignments: dict[tuple[str, int], dict[str, str]] = {}
    for pattern, expected in quotas.items():
        ordered = sorted(grouped[pattern], key=lambda value: (split_rank(pattern, value, seed), value))
        if len(ordered) != sum(expected.values()):
            raise ValueError(f"Group count mismatch for {pattern}: {len(ordered)} != {sum(expected.values())}")
        cursor = 0
        for partition in ("train", "validation", "test"):
            for local_index in ordered[cursor : cursor + expected[partition]]:
                assignments[(pattern, local_index)] = {
                    "partition": partition,
                    "rank_sha256": split_rank(pattern, local_index, seed),
                }
            cursor += expected[partition]
    return assignments


def _relative_dataset_id(path: Path) -> str:
    return path.resolve().relative_to(datasets_root()).as_posix()


def discover_annotated_records() -> list[dict[str, Any]]:
    """Read metadata for the 740 supervised images without touching ``final_320``."""

    image_root = dataset_path(*ANNOTATED_IMAGES_RELATIVE)
    annotation_root = dataset_path(*ANNOTATIONS_RELATIVE)
    records: list[dict[str, Any]] = []
    for run in RUNS:
        run_image_root = image_root / run
        run_annotation_root = annotation_root / run
        if not run_image_root.is_dir() or not run_annotation_root.is_dir():
            raise FileNotFoundError(f"Missing annotated run {run}")
        for image_path in sorted(run_image_root.glob("*.jpg"), key=lambda item: item.name.lower()):
            local_index, pattern = parse_annotated_stem(image_path.stem)
            annotation_path = run_annotation_root / f"{image_path.stem}.tsv"
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
            decoded = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise ValueError(f"Could not decode {image_path}")
            if decoded.shape != (512, 512):
                raise ValueError(f"Unexpected shape {decoded.shape} for {image_path}")
            records.append(
                {
                    "canonical_image_id": f"{run}/{image_path.stem}",
                    "run": run,
                    "pattern": pattern,
                    "local_index": local_index,
                    "image_source_id": _relative_dataset_id(image_path),
                    "annotation_source_id": _relative_dataset_id(annotation_path),
                    "image_sha256": sha256_file(image_path),
                    "annotation_sha256": sha256_file(annotation_path),
                    "width": 512,
                    "height": 512,
                }
            )
    if len(records) != 740:
        raise ValueError(f"Expected 740 annotated images, found {len(records)}")
    return records


def build_split_manifest(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    group_members: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_members[(str(record["pattern"]), int(record["local_index"]))].append(record)
    assignments = assign_grouped_split(group_members)
    images: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for key in sorted(group_members, key=lambda value: (PATTERNS.index(value[0]), value[1])):
        members = sorted(group_members[key], key=lambda item: item["run"])
        runs = [str(item["run"]) for item in members]
        if runs != list(RUNS):
            raise ValueError(f"Leakage group {key} has runs {runs}, expected {RUNS}")
        assignment = assignments[key]
        group_id = f"{key[0]}/{key[1]}"
        groups.append(
            {
                "group_id": group_id,
                "pattern": key[0],
                "local_index": key[1],
                "partition": assignment["partition"],
                "rank_sha256": assignment["rank_sha256"],
                "canonical_image_ids": [item["canonical_image_id"] for item in members],
            }
        )
        for member in members:
            images.append({**member, "group_id": group_id, "partition": assignment["partition"]})

    group_counts = Counter(item["partition"] for item in groups)
    image_counts = Counter(item["partition"] for item in images)
    by_pattern: dict[str, dict[str, int]] = {}
    for pattern in PATTERNS:
        by_pattern[pattern] = dict(
            Counter(item["partition"] for item in groups if item["pattern"] == pattern)
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "split_seed": SPLIT_SEED,
        "ranking_rule": "SHA256(seed|pattern|local_index), ascending",
        "partition_order": ["train", "validation", "test"],
        "group_definition": "pattern + local_index; all R1..R5 remain together",
        "expected_pattern_quotas": PATTERN_QUOTAS,
        "group_counts": dict(sorted(group_counts.items())),
        "image_counts": dict(sorted(image_counts.items())),
        "group_counts_by_pattern": by_pattern,
        "groups": groups,
        "images": sorted(images, key=lambda item: item["canonical_image_id"]),
    }
    manifest["content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def resolve_source_id(source_id: str) -> Path:
    parts = tuple(Path(source_id).parts)
    resolved = dataset_path(*parts)
    if datasets_root() not in (resolved, *resolved.parents):
        raise ValueError(f"Source escaped dataset root: {source_id}")
    return resolved


def read_annotations(path: Path) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["x", "y"]:
            raise ValueError(f"Unexpected TSV header in {path}: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            try:
                x_value = int(row["x"])
                y_value = int(row["y"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid annotation at {path}:{row_number}") from error
            if not (0 <= x_value < 512 and 0 <= y_value < 512):
                raise ValueError(f"Out-of-bounds annotation at {path}:{row_number}: {(x_value, y_value)}")
            points.append((x_value, y_value))
    return points


def deduplicate_points(points: Iterable[tuple[int, int]]) -> tuple[list[tuple[int, int]], int]:
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    duplicates = 0
    for point in points:
        normalized = (int(point[0]), int(point[1]))
        if normalized in seen:
            duplicates += 1
        else:
            seen.add(normalized)
            unique.append(normalized)
    return unique, duplicates


def build_ground_truth_summary(split_manifest: dict[str, Any]) -> dict[str, Any]:
    per_image: list[dict[str, Any]] = []
    totals = Counter()
    partition_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for record in split_manifest["images"]:
        points = read_annotations(resolve_source_id(record["annotation_source_id"]))
        unique, duplicates = deduplicate_points(points)
        row = {
            "canonical_image_id": record["canonical_image_id"],
            "partition": record["partition"],
            "before": len(points),
            "after": len(unique),
            "duplicates_removed": duplicates,
        }
        per_image.append(row)
        totals.update(before=len(points), after=len(unique), duplicates_removed=duplicates)
        partition_totals[record["partition"]].update(
            before=len(points), after=len(unique), duplicates_removed=duplicates
        )
    if totals["duplicates_removed"] != 241:
        raise ValueError(f"Expected exactly 241 duplicate records, found {totals['duplicates_removed']}")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "deduplication_key": "(canonical_image_id, x, y); preserve first occurrence",
        "totals": dict(totals),
        "partition_totals": {key: dict(value) for key, value in sorted(partition_totals.items())},
        "per_image": per_image,
    }


def robust_normalize_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got {gray.shape}")
    values = gray.astype(np.float32, copy=False)
    low, high = np.percentile(values, (1.0, 99.0))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip((values - low) * (255.0 / (high - low)), 0.0, 255.0).astype(np.uint8)


def preprocess_image(gray: np.ndarray) -> np.ndarray:
    normalized = robust_normalize_uint8(gray)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)
    return equalized.astype(np.float32) / np.float32(255.0)


def sample_affine_parameters(rng: np.random.Generator) -> dict[str, float]:
    return {
        "angle_degrees": float(rng.uniform(-7.0, 7.0)),
        "translate_x": float(rng.uniform(-16.0, 16.0)),
        "translate_y": float(rng.uniform(-16.0, 16.0)),
        "scale": float(rng.uniform(0.95, 1.05)),
        "contrast": float(rng.uniform(0.85, 1.15)),
        "brightness": float(rng.uniform(-0.10, 0.10)),
    }


def affine_matrix(width: int, height: int, parameters: dict[str, float]) -> np.ndarray:
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, parameters["angle_degrees"], parameters["scale"]).astype(np.float64)
    matrix[0, 2] += parameters["translate_x"]
    matrix[1, 2] += parameters["translate_y"]
    return matrix


def transform_points(
    points: np.ndarray | Sequence[Sequence[float]],
    matrix: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(array) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)
    homogeneous = np.column_stack([array, np.ones(len(array), dtype=np.float64)])
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    valid = (
        (transformed[:, 0] >= 0.0)
        & (transformed[:, 0] <= width - 1)
        & (transformed[:, 1] >= 0.0)
        & (transformed[:, 1] <= height - 1)
    )
    return transformed[valid].astype(np.float32), valid


def augment_image_and_points(
    image: np.ndarray,
    points: np.ndarray,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    matrix = affine_matrix(width, height, parameters)
    warped = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed, _ = transform_points(points, matrix, width=width, height=height)
    adjusted = (warped - 0.5) * parameters["contrast"] + 0.5 + parameters["brightness"]
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32), transformed, matrix


def gaussian_target(
    shape: tuple[int, int],
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    sigma: float = 1.5,
    truncate: float = 4.0,
) -> np.ndarray:
    if sigma <= 0 or truncate <= 0:
        raise ValueError("sigma and truncate must be positive")
    height, width = shape
    target = np.zeros((height, width), dtype=np.float32)
    radius = int(math.ceil(sigma * truncate))
    for x_value, y_value in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        x_start = max(0, int(math.floor(x_value)) - radius)
        x_stop = min(width - 1, int(math.ceil(x_value)) + radius)
        y_start = max(0, int(math.floor(y_value)) - radius)
        y_stop = min(height - 1, int(math.ceil(y_value)) + radius)
        if x_start > x_stop or y_start > y_stop:
            continue
        x_grid = np.arange(x_start, x_stop + 1, dtype=np.float64)
        y_grid = np.arange(y_start, y_stop + 1, dtype=np.float64)
        squared = (x_grid[None, :] - x_value) ** 2 + (y_grid[:, None] - y_value) ** 2
        patch = np.exp(-squared / (2.0 * sigma * sigma)).astype(np.float32)
        view = target[y_start : y_stop + 1, x_start : x_stop + 1]
        np.maximum(view, patch, out=view)
    return target


@dataclass(frozen=True)
class PeakSet:
    coordinates: np.ndarray
    scores: np.ndarray


def _plateau_representatives(heatmap: np.ndarray, candidate_mask: np.ndarray) -> list[tuple[float, int, int]]:
    height, width = heatmap.shape
    visited = np.zeros(candidate_mask.shape, dtype=bool)
    representatives: list[tuple[float, int, int]] = []
    for y_value, x_value in np.argwhere(candidate_mask):
        y_int, x_int = int(y_value), int(x_value)
        if visited[y_int, x_int]:
            continue
        score = float(heatmap[y_int, x_int])
        queue: deque[tuple[int, int]] = deque([(y_int, x_int)])
        visited[y_int, x_int] = True
        plateau: list[tuple[int, int]] = []
        while queue:
            current_y, current_x = queue.popleft()
            plateau.append((current_y, current_x))
            for neighbor_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                for neighbor_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                    if visited[neighbor_y, neighbor_x] or not candidate_mask[neighbor_y, neighbor_x]:
                        continue
                    if float(heatmap[neighbor_y, neighbor_x]) == score:
                        visited[neighbor_y, neighbor_x] = True
                        queue.append((neighbor_y, neighbor_x))
        chosen_y, chosen_x = min(plateau)
        representatives.append((score, chosen_y, chosen_x))
    return representatives


def extract_peaks(heatmap: np.ndarray, *, threshold: float, nms_radius: float) -> PeakSet:
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be two-dimensional")
    if nms_radius < 0:
        raise ValueError("nms_radius cannot be negative")
    values = np.asarray(heatmap, dtype=np.float32)
    local_maximum = values == cv2.dilate(values, np.ones((3, 3), dtype=np.uint8))
    candidates = local_maximum & np.isfinite(values) & (values >= float(threshold))
    ordered = sorted(_plateau_representatives(values, candidates), key=lambda row: (-row[0], row[1], row[2]))
    accepted: list[tuple[float, int, int]] = []
    squared_radius = float(nms_radius) ** 2
    for candidate in ordered:
        _, y_value, x_value = candidate
        if any((x_value - other_x) ** 2 + (y_value - other_y) ** 2 <= squared_radius for _, other_y, other_x in accepted):
            continue
        accepted.append(candidate)
    coordinates = np.asarray([(x_value, y_value) for _, y_value, x_value in accepted], dtype=np.float32).reshape(-1, 2)
    scores = np.asarray([score for score, _, _ in accepted], dtype=np.float32)
    return PeakSet(coordinates=coordinates, scores=scores)


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: float


def _add_flow_edge(graph: list[list[_FlowEdge]], source: int, target: int, capacity: int, cost: float) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity, cost)
    reverse = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def optimal_point_matching(
    predictions: np.ndarray | Sequence[Sequence[float]],
    ground_truth: np.ndarray | Sequence[Sequence[float]],
    tolerance: float,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, minimum-total-distance bipartite matching."""

    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1, 2)
    truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 2)
    if len(predicted) == 0 or len(truth) == 0:
        return []

    prediction_count = len(predicted)
    truth_count = len(truth)
    source = 0
    prediction_offset = 1
    truth_offset = prediction_offset + prediction_count
    sink = truth_offset + truth_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    for prediction_index in range(prediction_count):
        _add_flow_edge(graph, source, prediction_offset + prediction_index, 1, 0.0)
    for truth_index in range(truth_count):
        _add_flow_edge(graph, truth_offset + truth_index, sink, 1, 0.0)

    edge_references: list[tuple[int, int, float, _FlowEdge]] = []
    for prediction_index, coordinate in enumerate(predicted):
        distances = np.sqrt(np.sum((truth - coordinate) ** 2, axis=1))
        for truth_index in np.flatnonzero(distances <= tolerance + 1e-12):
            distance = float(distances[truth_index])
            edge = _add_flow_edge(
                graph,
                prediction_offset + prediction_index,
                truth_offset + int(truth_index),
                1,
                distance,
            )
            edge_references.append((prediction_index, int(truth_index), distance, edge))

    node_count = len(graph)
    potential = [0.0] * node_count
    infinity = float("inf")
    while True:
        distance = [infinity] * node_count
        previous_node = [-1] * node_count
        previous_edge = [-1] * node_count
        distance[source] = 0.0
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance > distance[node] + 1e-12:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                reduced = edge.cost + potential[node] - potential[edge.to]
                proposal = current_distance + reduced
                if proposal + 1e-12 < distance[edge.to]:
                    distance[edge.to] = proposal
                    previous_node[edge.to] = node
                    previous_edge[edge.to] = edge_index
                    heapq.heappush(queue, (proposal, edge.to))
        if not math.isfinite(distance[sink]):
            break
        for node in range(node_count):
            if math.isfinite(distance[node]):
                potential[node] += distance[node]
        node = sink
        while node != source:
            parent = previous_node[node]
            edge_index = previous_edge[node]
            if parent < 0 or edge_index < 0:
                raise RuntimeError("Broken augmenting path")
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent

    matches = [
        (prediction_index, truth_index, distance)
        for prediction_index, truth_index, distance, edge in edge_references
        if edge.capacity == 0
    ]
    return sorted(matches)


def point_metrics(
    predictions: np.ndarray | Sequence[Sequence[float]],
    ground_truth: np.ndarray | Sequence[Sequence[float]],
    tolerance: float,
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1, 2)
    truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 2)
    matches = optimal_point_matching(predicted, truth, tolerance)
    true_positives = len(matches)
    false_positives = len(predicted) - true_positives
    false_negatives = len(truth) - true_positives
    precision = true_positives / len(predicted) if len(predicted) else (1.0 if len(truth) == 0 else 0.0)
    recall = true_positives / len(truth) if len(truth) else (1.0 if len(predicted) == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tolerance": float(tolerance),
        "predictions": int(len(predicted)),
        "ground_truth": int(len(truth)),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_localization_error": float(np.mean([item[2] for item in matches])) if matches else None,
        "matches": [
            {"prediction_index": item[0], "ground_truth_index": item[1], "distance": item[2]} for item in matches
        ],
    }


def threshold_cardinality_curve(
    coordinates: np.ndarray,
    scores: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: Sequence[float],
    *,
    tolerance: float,
) -> dict[float, dict[str, int]]:
    """Compute exact TP counts for nested score thresholds incrementally.

    Only maximum cardinality is needed while selecting a threshold. The final
    selected operating point is still scored with minimum-distance matching.
    """

    predicted = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    confidence = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 2)
    if len(predicted) != len(confidence):
        raise ValueError("Coordinate and score counts differ")
    order = np.lexsort((predicted[:, 0], predicted[:, 1], -confidence)) if len(predicted) else np.empty(0, int)
    predicted = predicted[order]
    confidence = confidence[order]
    adjacency: list[list[int]] = []
    for coordinate in predicted:
        distances = np.sqrt(np.sum((truth - coordinate) ** 2, axis=1)) if len(truth) else np.empty(0)
        adjacency.append(sorted(np.flatnonzero(distances <= tolerance + 1e-12), key=lambda index: (distances[index], index)))

    matched_prediction_for_truth = [-1] * len(truth)
    active_count = 0
    matched_count = 0

    def augment(prediction_index: int, seen_predictions: set[int], seen_truth: set[int]) -> bool:
        if prediction_index in seen_predictions:
            return False
        seen_predictions.add(prediction_index)
        for truth_index in adjacency[prediction_index]:
            if truth_index in seen_truth:
                continue
            seen_truth.add(truth_index)
            previous = matched_prediction_for_truth[truth_index]
            if previous < 0 or augment(previous, seen_predictions, seen_truth):
                matched_prediction_for_truth[truth_index] = prediction_index
                return True
        return False

    output: dict[float, dict[str, int]] = {}
    for threshold in sorted({float(value) for value in thresholds}, reverse=True):
        while active_count < len(predicted) and confidence[active_count] >= threshold:
            if augment(active_count, set(), set()):
                matched_count += 1
            active_count += 1
        output[threshold] = {
            "predictions": active_count,
            "ground_truth": len(truth),
            "true_positives": matched_count,
            "false_positives": active_count - matched_count,
            "false_negatives": len(truth) - matched_count,
        }
    return output


def aggregate_point_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    true_positives = sum(int(row["true_positives"]) for row in rows)
    false_positives = sum(int(row["false_positives"]) for row in rows)
    false_negatives = sum(int(row["false_negatives"]) for row in rows)
    prediction_count = true_positives + false_positives
    truth_count = true_positives + false_negatives
    precision = true_positives / prediction_count if prediction_count else (1.0 if truth_count == 0 else 0.0)
    recall = true_positives / truth_count if truth_count else (1.0 if prediction_count == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    distances = [match["distance"] for row in rows for match in row.get("matches", [])]
    return {
        "images": len(rows),
        "predictions": prediction_count,
        "ground_truth": truth_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_localization_error": float(np.mean(distances)) if distances else None,
        "false_positives_per_image": float(false_positives / len(rows)) if rows else None,
        "mean_recall_per_image": float(np.mean([row["recall"] for row in rows])) if rows else None,
    }


def baseline_response(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    values = np.asarray(image, dtype=np.float32)
    blurred = cv2.GaussianBlur(values, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return values - blurred


def baseline_detect(image: np.ndarray, *, sigma: float, percentile: float, nms_radius: float) -> PeakSet:
    response = baseline_response(image, sigma)
    threshold = float(np.percentile(response, percentile))
    return extract_peaks(response, threshold=threshold, nms_radius=nms_radius)


def tile_origins(length: int, *, tile_size: int = 512, overlap: int = 64) -> list[int]:
    if length <= 0 or tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Invalid tiling dimensions")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    origins = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if origins[-1] != final:
        origins.append(final)
    return origins


def cosine_blend_weight(
    tile_size: int = 512,
    ramp: int = 32,
    *,
    at_top: bool = False,
    at_bottom: bool = False,
    at_left: bool = False,
    at_right: bool = False,
) -> np.ndarray:
    if ramp < 0 or ramp * 2 >= tile_size:
        raise ValueError("Invalid blend ramp")
    one_dimensional = np.ones(tile_size, dtype=np.float32)
    if ramp:
        phase = (np.arange(ramp, dtype=np.float32) + 0.5) / ramp
        edge = 0.5 - 0.5 * np.cos(np.pi * phase)
        one_dimensional[:ramp] = edge
        one_dimensional[-ramp:] = edge[::-1]
    vertical = one_dimensional.copy()
    horizontal = one_dimensional.copy()
    if at_top:
        vertical[:ramp] = 1.0
    if at_bottom:
        vertical[-ramp:] = 1.0
    if at_left:
        horizontal[:ramp] = 1.0
    if at_right:
        horizontal[-ramp:] = 1.0
    return vertical[:, None] * horizontal[None, :]


def blend_tiles(
    shape: tuple[int, int],
    tiles: Sequence[tuple[int, int, np.ndarray]],
    *,
    tile_size: int = 512,
    ramp: int = 32,
) -> np.ndarray:
    height, width = shape
    accumulator = np.zeros(shape, dtype=np.float64)
    weights = np.zeros(shape, dtype=np.float64)
    for y_value, x_value, tile in tiles:
        tile_array = np.asarray(tile, dtype=np.float32)
        if tile_array.shape != (tile_size, tile_size):
            raise ValueError(f"Unexpected tile shape {tile_array.shape}")
        valid_height = min(tile_size, height - y_value)
        valid_width = min(tile_size, width - x_value)
        weight = cosine_blend_weight(
            tile_size,
            ramp,
            at_top=y_value == 0,
            at_bottom=y_value + tile_size >= height,
            at_left=x_value == 0,
            at_right=x_value + tile_size >= width,
        )[:valid_height, :valid_width]
        accumulator[y_value : y_value + valid_height, x_value : x_value + valid_width] += (
            tile_array[:valid_height, :valid_width] * weight
        )
        weights[y_value : y_value + valid_height, x_value : x_value + valid_width] += weight
    if np.any(weights <= 0):
        raise ValueError("Tiles do not cover the requested output shape")
    return (accumulator / weights).astype(np.float32)


def repeatability(matched: int, count_first: int, count_second: int) -> dict[str, Any]:
    if min(matched, count_first, count_second) < 0 or matched > min(count_first, count_second):
        raise ValueError("Invalid repeatability counts")
    denominator = count_first + count_second
    if denominator == 0:
        return {"status": "ZERO_DETECTIONS_BOTH", "repeatability": None}
    return {"status": "OK", "repeatability": float(2.0 * matched / denominator)}


def paired_bootstrap_median(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Bootstrap values must be a non-empty finite vector")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(resamples, len(array)))
    statistics = np.median(array[indices], axis=1)
    lower, upper = np.percentile(statistics, (2.5, 97.5))
    return {
        "statistic": "median",
        "estimate": float(np.median(array)),
        "confidence_level": 0.95,
        "method": "paired percentile bootstrap",
        "resamples": int(resamples),
        "seed": int(seed),
        "lower": float(lower),
        "upper": float(upper),
    }


RESUME_STATE_KEYS = (
    "state_dict",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "best_validation_loss",
    "best_epoch",
    "early_stop_best",
    "epochs_without_improvement",
    "elapsed_seconds",
    "history",
)


def validate_resume_state(state: Any, *, seed: int, model_configuration: Any, require_full: bool = True) -> Any:
    """Reject a training resume state that does not continue this exact frozen run.

    Silently resuming from another seed, another frozen configuration, or a
    truncated state would change the trained model, so every mismatch raises.
    """

    if not isinstance(state, dict):
        raise RuntimeError("Resume state is not a mapping")
    if state.get("schema_version") != 1:
        raise RuntimeError("Resume state has an unsupported schema version")
    if int(state.get("seed", -1)) != int(seed):
        raise RuntimeError(f"Resume state belongs to seed {state.get('seed')}, not {seed}")
    if state.get("model_configuration") != model_configuration:
        raise RuntimeError("Resume state uses a different frozen configuration")
    epoch = state.get("epoch")
    if not isinstance(epoch, int) or not 1 <= epoch <= int(model_configuration["maximum_epochs"]):
        raise RuntimeError(f"Resume state records an out-of-range epoch: {epoch!r}")
    if require_full:
        missing = [key for key in RESUME_STATE_KEYS if key not in state]
        if missing:
            raise RuntimeError(f"Resume state is incomplete; missing {missing}")
    return state


def gate_a_decision(inputs: dict[str, Any]) -> str:
    strong = (
        inputs["median_f1_4px"] >= 0.80
        and inputs["median_precision_4px"] >= 0.75
        and inputs["median_recall_4px"] >= 0.75
        and inputs["absolute_f1_advantage"] >= 0.10
        and all(value >= 0.70 for value in inputs["run_median_f1_4px"].values())
        and inputs["seed_f1_spread"] <= 0.05
    )
    conditional = (
        inputs["median_f1_4px"] >= 0.65
        and inputs["all_seeds_above_baseline"]
        and not strong
    )
    return "STRONG_PASS" if strong else ("CONDITIONAL" if conditional else "FAIL")


def blinded_gate_b_decision(
    *,
    mated_valid_registrations: int,
    paired_fingers: int,
    median_delta: float | None,
    bootstrap_lower: float | None,
    positive_seed_count: int,
) -> dict[str, Any]:
    conditions = {
        "at_least_15_mated_valid": mated_valid_registrations >= 15,
        "median_delta_at_least_0_10": median_delta is not None and median_delta >= 0.10,
        "bootstrap_lower_above_zero": bootstrap_lower is not None and bootstrap_lower > 0,
        "positive_direction_at_least_two_seeds": positive_seed_count >= 2,
    }
    sufficient_paired = paired_fingers >= 15
    inconsistent_direction = positive_seed_count < 2
    if sufficient_paired and (median_delta is None or median_delta <= 0 or inconsistent_direction):
        outcome = "TRANSFER_FAIL"
    elif all(conditions.values()):
        outcome = "PENDING_EXPERIMENT_001_UNBLIND"
    else:
        outcome = "TRANSFER_INCONCLUSIVE"
    return {
        "conditions_1_to_4": conditions,
        "positive_seed_count": positive_seed_count,
        "sufficient_paired_fingers": sufficient_paired,
        "outcome_before_unblind": outcome,
    }


def edge_mask(points: np.ndarray | Sequence[Sequence[float]], *, width: int = 512, height: int = 512, band: int = 8) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(array) == 0:
        return np.empty((0,), dtype=bool)
    x_values, y_values = array[:, 0], array[:, 1]
    return np.minimum.reduce([x_values, y_values, width - 1 - x_values, height - 1 - y_values]) <= band


def ridge_area_mask(image: np.ndarray) -> np.ndarray:
    uint8 = np.clip(np.asarray(image) * 255.0 if image.dtype.kind == "f" else image, 0, 255).astype(np.uint8)
    _, inverted = cv2.threshold(uint8, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(
        inverted,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((dilated > 0).astype(np.uint8), connectivity=8)
    minimum_area = 0.01 * uint8.size
    mask = np.zeros(uint8.shape, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            mask |= labels == label
    return mask
