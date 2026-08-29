"""Deterministic primitives used by Experiment 003.

The functions in this module are dataset-independent.  Dataset traversal and
image matching remain in experiment scripts so CI does not require source data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

FINAL_FILENAME_RE = re.compile(r"^(?P<identity>[1-9]\d*)_(?P<capture_group>[12])_(?P<instance>[1-5])\.png$")
ANNOTATED_FILENAME_RE = re.compile(
    r"^(?P<local_index>[1-9]\d*)_(?P<pattern>left_loop|plain_arch|right_loop|tented_arch|whorl)\.jpg$"
)


@dataclass(frozen=True, slots=True)
class FinalSampleName:
    """Parsed fields from one final L3-SF filename."""

    identity: int
    capture_group: int
    instance: int


@dataclass(frozen=True, slots=True)
class AnnotatedSampleName:
    """Parsed fields from one annotated L3-SF filename."""

    local_index: int
    pattern: str


def parse_final_filename(filename: str) -> FinalSampleName:
    """Parse an exact final-image basename or raise ``ValueError``."""

    match = FINAL_FILENAME_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"Invalid final_320 filename: {filename!r}")
    return FinalSampleName(
        identity=int(match.group("identity")),
        capture_group=int(match.group("capture_group")),
        instance=int(match.group("instance")),
    )


def parse_annotated_filename(filename: str) -> AnnotatedSampleName:
    """Parse an exact annotated-image basename or raise ``ValueError``."""

    match = ANNOTATED_FILENAME_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"Invalid annotated_512 filename: {filename!r}")
    return AnnotatedSampleName(
        local_index=int(match.group("local_index")),
        pattern=match.group("pattern"),
    )


def deterministic_rank(seed: str, *parts: object) -> str:
    """Return the SHA-256 rank used for frozen deterministic sampling."""

    payload = "|".join([seed, *(str(part) for part in parts)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_stratified(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    quotas: Mapping[tuple[str, str], int],
    id_field: str = "canonical_annotated_id",
    run_field: str = "run",
    pattern_field: str = "pattern",
) -> list[dict[str, Any]]:
    """Select deterministic per-(run, pattern) quotas without replacement."""

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for (run, pattern), quota in sorted(quotas.items()):
        if quota < 0:
            raise ValueError("Quotas must be non-negative")
        candidates = [
            dict(record)
            for record in records
            if record.get(run_field) == run and record.get(pattern_field) == pattern
        ]
        candidates.sort(key=lambda row: (deterministic_rank(seed, run, row[id_field]), str(row[id_field])))
        if len(candidates) < quota:
            raise ValueError(f"Insufficient records for {(run, pattern)}: {len(candidates)} < {quota}")
        for candidate in candidates[:quota]:
            canonical_id = str(candidate[id_field])
            if canonical_id in seen_ids:
                raise ValueError(f"Duplicate selected identifier: {canonical_id}")
            seen_ids.add(canonical_id)
            candidate["selection_rank_sha256"] = deterministic_rank(seed, run, canonical_id)
            candidate["stratum_quota"] = quota
            selected.append(candidate)
    return sorted(selected, key=lambda row: (str(row[run_field]), str(row[pattern_field]), row["selection_rank_sha256"]))


def deduplicate_points(points: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """Remove exact point duplicates while preserving first-occurrence order."""

    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for point in points:
        if len(point) != 2:
            raise ValueError(f"Expected a 2-D point, got {point!r}")
        normalized = (int(point[0]), int(point[1]))
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def transform_points(points: Sequence[Sequence[float]], matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Apply a 2x3 affine or 3x3 projective transform to 2-D points."""

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("Points must have shape (n, 2)")
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape == (2, 3):
        x_values = point_array[:, 0]
        y_values = point_array[:, 1]
        return np.column_stack(
            [
                transform[0, 0] * x_values + transform[0, 1] * y_values + transform[0, 2],
                transform[1, 0] * x_values + transform[1, 1] * y_values + transform[1, 2],
            ]
        )
    if transform.shape != (3, 3):
        raise ValueError("Transform must have shape (2, 3) or (3, 3)")
    x_values = point_array[:, 0]
    y_values = point_array[:, 1]
    projected_x = transform[0, 0] * x_values + transform[0, 1] * y_values + transform[0, 2]
    projected_y = transform[1, 0] * x_values + transform[1, 1] * y_values + transform[1, 2]
    denominators = transform[2, 0] * x_values + transform[2, 1] * y_values + transform[2, 2]
    if np.any(np.isclose(denominators, 0.0)):
        raise ValueError("Projective transform maps a point to infinity")
    return np.column_stack([projected_x / denominators, projected_y / denominators])


def effective_affine_scale(matrix: Sequence[Sequence[float]]) -> float:
    """Return the area-equivalent linear scale of an affine transform."""

    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape not in {(2, 3), (3, 3)}:
        raise ValueError("Transform must have shape (2, 3) or (3, 3)")
    determinant = float(transform[0, 0] * transform[1, 1] - transform[0, 1] * transform[1, 0])
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("Transform has a non-finite or zero linear determinant")
    return math.sqrt(abs(determinant))


def round_half_up(value: float) -> int:
    """Round a non-negative finite value with halves away from zero."""

    if not math.isfinite(value) or value < 0:
        raise ValueError("Value must be finite and non-negative")
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def scaled_tolerance(matrix: Sequence[Sequence[float]], base_radius: float = 4.0) -> int:
    """Convert an annotated-coordinate tolerance to final-image pixels."""

    if not math.isfinite(base_radius) or base_radius <= 0:
        raise ValueError("Base radius must be finite and positive")
    return max(1, round_half_up(base_radius * effective_affine_scale(matrix)))


def maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> list[int]:
    """Return a deterministic maximum-weight row-to-column assignment.

    This is the O(n^3) Hungarian algorithm for matrices with no more rows than
    columns.  Ties retain the lowest available column index.
    """

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Weights must be a two-dimensional matrix")
    row_count, column_count = values.shape
    if row_count == 0:
        return []
    if row_count > column_count:
        raise ValueError("Assignment requires row_count <= column_count")
    if not np.all(np.isfinite(values)):
        raise ValueError("Weights must be finite")

    costs = float(np.max(values)) - values
    u = np.zeros(row_count + 1, dtype=np.float64)
    v = np.zeros(column_count + 1, dtype=np.float64)
    p = np.zeros(column_count + 1, dtype=np.int64)
    way = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        p[0] = row
        min_values = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = math.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_values[column]:
                    min_values[column] = current
                    way[column] = column0
                if min_values[column] < delta:
                    delta = float(min_values[column])
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if p[column] != 0:
            assignment[int(p[column]) - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("Hungarian assignment did not cover every row")
    return assignment


def write_json(path: Path, payload: Any) -> None:
    """Write canonical, human-readable JSON with a terminal newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    """Write deterministic UTF-8 CSV using an explicit or sorted schema."""

    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dataclass_dict(value: Any) -> dict[str, Any]:
    """Expose a typed parsed-name dataclass as a serializable dictionary."""

    return asdict(value)
