#!/usr/bin/env python3
"""Inventory local L3-SF, validate pore annotations, and freeze audit sample."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from fingerprint_new_method.paths import dataset_path

WORK_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = dataset_path("L3_SF_V2", "L3SF_V2")
FULL_ROOT = DATA_ROOT / "L3-SF"
ANN_IMAGE_ROOT = DATA_ROOT / "Pore ground truth" / "Fingerprint Images"
ANN_ROOT = DATA_ROOT / "Pore ground truth" / "Ground truth"
OUT_ROOT = WORK_ROOT / "artifacts" / "experiment-002"
INVENTORY_ROOT = OUT_ROOT / "inventory"
REVIEW_ROOT = OUT_ROOT / "review"

AUDIT_SEED = "l3sf-exp002-blind-audit-v1"
PATTERN_QUOTAS_PER_RUN = {
    "right_loop": 4,
    "whorl": 3,
    "left_loop": 1,
    "plain_arch": 1,
    "tented_arch": 1,
}

# These annotated images were displayed only to resolve the on-disk structure
# before the blind audit sample was frozen. They remain in all inventories and
# integrity/spatial analyses but are ineligible for the 50-image visual audit.
# This is a blindness exclusion, never an image-quality exclusion.
PRE_FREEZE_MAPPING_PILOT = {
    "R1/1_left_loop",
    "R1/1_plain_arch",
    "R1/1_right_loop",
    "R1/1_whorl",
    "R1/18_right_loop",
    "R1/44_right_loop",
    "R2/1_whorl",
    "R3/1_whorl",
    "R4/1_whorl",
    "R5/1_whorl",
}

FULL_RE = re.compile(r"^(?P<identity>\d+)_(?P<capture_group>\d+)_(?P<instance>\d+)\.png$", re.I)
ANN_RE = re.compile(
    r"^(?P<local_index>\d+)_(?P<pattern>left_loop|plain_arch|right_loop|tented_arch|whorl)$",
    re.I,
)


def rel(path: Path) -> str:
    return path.relative_to(DATA_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_hex(*parts: object) -> str:
    payload = "|".join(str(part) for part in (AUDIT_SEED, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def image_record(path: Path, subset: str, run: str, parsed: dict[str, Any]) -> dict[str, Any]:
    image = load_image(path)
    record: dict[str, Any] = {
        "subset": subset,
        "run": run,
        "relative_path": rel(path),
        "filename": path.name,
        "format": path.suffix.lower().lstrip("."),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **parsed,
    }
    if image is None:
        record.update({"decode_status": "FAIL", "width": None, "height": None, "channels": None})
    else:
        channels = 1 if image.ndim == 2 else image.shape[2]
        record.update(
            {
                "decode_status": "PASS",
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "channels": int(channels),
            }
        )
    return record


def parse_tsv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = [field.strip() for field in (reader.fieldnames or [])]
            if fields != ["x", "y"]:
                errors.append(f"unexpected_fields:{fields}")
            for row_index, raw in enumerate(reader, start=1):
                try:
                    x_raw = raw.get("x")
                    y_raw = raw.get("y")
                    x_float = float(x_raw) if x_raw is not None else math.nan
                    y_float = float(y_raw) if y_raw is not None else math.nan
                    if not x_float.is_integer() or not y_float.is_integer():
                        errors.append(f"non_integer_coordinate:row={row_index}:x={x_raw}:y={y_raw}")
                    rows.append({"row_index": row_index, "x": int(x_float), "y": int(y_float)})
                except (TypeError, ValueError, OverflowError) as exc:
                    errors.append(f"parse_error:row={row_index}:{exc}")
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"file_error:{exc}")
    return rows, errors


def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.full(len(points), np.nan, dtype=np.float64)
    result = np.empty(len(points), dtype=np.float64)
    chunk_size = 256
    for start in range(0, len(points), chunk_size):
        stop = min(len(points), start + chunk_size)
        diff = points[start:stop, None, :] - points[None, :, :]
        distances_sq = np.sum(diff * diff, axis=2)
        for local, global_index in enumerate(range(start, stop)):
            distances_sq[local, global_index] = np.inf
        result[start:stop] = np.sqrt(np.min(distances_sq, axis=1))
    return result


def estimated_fingerprint_mask(gray: np.ndarray) -> tuple[np.ndarray, float]:
    threshold, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    connected = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    connected = cv2.dilate(connected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    minimum_component_area = gray.size * 0.01
    retained = [contour for contour in contours if cv2.contourArea(contour) >= minimum_component_area]
    if retained:
        cv2.drawContours(mask, retained, -1, 255, thickness=cv2.FILLED)
    else:
        mask[:] = 255
    return mask, float(threshold)


def distribution(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "minimum": None, "p05": None, "median": None, "p95": None, "maximum": None, "mean": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": len(finite),
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    if not DATA_ROOT.is_dir():
        raise RuntimeError(f"L3-SF root not found: {DATA_ROOT}")

    image_rows: list[dict[str, Any]] = []
    full_parse_failures: list[str] = []
    for run_dir in sorted(FULL_ROOT.glob("R*"), key=lambda path: path.name):
        for path in sorted(run_dir.glob("*.png"), key=lambda item: item.name):
            match = FULL_RE.match(path.name)
            parsed: dict[str, Any] = {}
            if match:
                parsed = {
                    "identity_number": int(match.group("identity")),
                    "capture_group": int(match.group("capture_group")),
                    "instance": int(match.group("instance")),
                    "canonical_identity": f"{run_dir.name}/{int(match.group('identity'))}",
                    "canonical_sample_id": f"{run_dir.name}/{path.stem}",
                }
            else:
                full_parse_failures.append(rel(path))
            image_rows.append(image_record(path, "full_l3sf", run_dir.name, parsed))

    annotation_image_map: dict[str, dict[str, Any]] = {}
    ann_name_parse_failures: list[str] = []
    for run_dir in sorted(ANN_IMAGE_ROOT.glob("R*"), key=lambda path: path.name):
        for path in sorted(run_dir.glob("*.jpg"), key=lambda item: item.name):
            match = ANN_RE.match(path.stem)
            parsed = {}
            if match:
                pattern = match.group("pattern").lower()
                local_index = int(match.group("local_index"))
                parsed = {
                    "annotation_local_index": local_index,
                    "pattern": pattern,
                    "canonical_identity": f"{run_dir.name}/{path.stem}",
                    "canonical_sample_id": f"{run_dir.name}/{path.stem}",
                }
            else:
                ann_name_parse_failures.append(rel(path))
            record = image_record(path, "annotated_master", run_dir.name, parsed)
            image_rows.append(record)
            annotation_image_map[f"{run_dir.name}/{path.stem}"] = record

    image_hash_groups: dict[str, list[str]] = defaultdict(list)
    for row in image_rows:
        image_hash_groups[row["sha256"]].append(row["relative_path"])
    duplicate_image_groups = [members for members in image_hash_groups.values() if len(members) > 1]

    annotation_rows: list[dict[str, Any]] = []
    annotation_file_rows: list[dict[str, Any]] = []
    orphan_tsv: list[str] = []
    malformed_tsv: list[dict[str, Any]] = []
    ann_hash_groups: dict[str, list[str]] = defaultdict(list)
    all_x_norm: list[float] = []
    all_y_norm: list[float] = []
    all_nn: list[float] = []
    per_image_spatial: list[dict[str, Any]] = []

    for run_dir in sorted(ANN_ROOT.glob("R*"), key=lambda path: path.name):
        for path in sorted(run_dir.glob("*.tsv"), key=lambda item: item.name):
            canonical = f"{run_dir.name}/{path.stem}"
            image_record_for_tsv = annotation_image_map.get(canonical)
            if image_record_for_tsv is None:
                orphan_tsv.append(rel(path))
                continue
            rows, errors = parse_tsv(path)
            if errors:
                malformed_tsv.append({"relative_path": rel(path), "errors": errors})
            width = int(image_record_for_tsv["width"] or 0)
            height = int(image_record_for_tsv["height"] or 0)
            coords = Counter((row["x"], row["y"]) for row in rows)
            duplicate_records = sum(count - 1 for count in coords.values() if count > 1)
            out_of_bounds = sum(not (0 <= row["x"] < width and 0 <= row["y"] < height) for row in rows)
            annotation_hash = sha256_file(path)
            ann_hash_groups[annotation_hash].append(rel(path))

            image_path = DATA_ROOT / image_record_for_tsv["relative_path"]
            image = load_image(image_path)
            if image is None:
                mask = np.zeros((height, width), dtype=np.uint8)
                otsu = math.nan
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                mask, otsu = estimated_fingerprint_mask(gray)
            points = np.asarray([(row["x"], row["y"]) for row in rows], dtype=np.float64)
            nn = nearest_neighbor_distances(points)
            mask_area = int(np.count_nonzero(mask))
            inside_mask = 0
            edge16 = 0
            edge5pct = 0
            for row, nn_distance in zip(rows, nn):
                x, y = row["x"], row["y"]
                in_bounds = 0 <= x < width and 0 <= y < height
                inside = bool(in_bounds and mask[y, x] != 0)
                inside_mask += int(inside)
                distance_to_edge = min(x, y, width - 1 - x, height - 1 - y) if in_bounds else -1
                edge16 += int(distance_to_edge >= 0 and distance_to_edge <= 16)
                edge5pct += int(distance_to_edge >= 0 and distance_to_edge <= 0.05 * min(width, height))
                x_norm = x / (width - 1) if width > 1 else math.nan
                y_norm = y / (height - 1) if height > 1 else math.nan
                if math.isfinite(x_norm):
                    all_x_norm.append(x_norm)
                if math.isfinite(y_norm):
                    all_y_norm.append(y_norm)
                if math.isfinite(float(nn_distance)):
                    all_nn.append(float(nn_distance))
                annotation_rows.append(
                    {
                        "canonical_image_id": canonical,
                        "run": run_dir.name,
                        "pattern": image_record_for_tsv["pattern"],
                        "annotation_local_index": image_record_for_tsv["annotation_local_index"],
                        "image_relative_path": image_record_for_tsv["relative_path"],
                        "annotation_relative_path": rel(path),
                        "annotation_row_index": row["row_index"],
                        "x": x,
                        "y": y,
                        "x_normalized": x_norm,
                        "y_normalized": y_norm,
                        "in_bounds": in_bounds,
                        "inside_estimated_fingerprint_mask": inside,
                        "distance_to_image_edge_px": distance_to_edge,
                        "nearest_neighbor_distance_px": float(nn_distance),
                    }
                )

            per_image = {
                "canonical_image_id": canonical,
                "run": run_dir.name,
                "pattern": image_record_for_tsv["pattern"],
                "image_relative_path": image_record_for_tsv["relative_path"],
                "annotation_relative_path": rel(path),
                "image_sha256": image_record_for_tsv["sha256"],
                "annotation_sha256": annotation_hash,
                "width": width,
                "height": height,
                "annotation_count": len(rows),
                "out_of_bounds_count": out_of_bounds,
                "duplicate_coordinate_record_count": duplicate_records,
                "estimated_fingerprint_area_px": mask_area,
                "estimated_fingerprint_area_fraction": mask_area / (width * height) if width and height else math.nan,
                "annotation_density_per_100k_estimated_fingerprint_px": len(rows) / mask_area * 100000 if mask_area else math.nan,
                "inside_estimated_fingerprint_mask_count": inside_mask,
                "within_16px_edge_count": edge16,
                "within_5pct_edge_count": edge5pct,
                "nearest_neighbor_median_px": float(np.nanmedian(nn)) if len(nn) else math.nan,
                "otsu_dark_ridge_threshold": otsu,
            }
            per_image_spatial.append(per_image)
            annotation_file_rows.append(per_image)

    expected_image_ids = set(annotation_image_map)
    observed_tsv_ids = {row["canonical_image_id"] for row in annotation_file_rows}
    images_without_tsv = sorted(expected_image_ids - observed_tsv_ids)
    duplicate_annotation_file_groups = [members for members in ann_hash_groups.values() if len(members) > 1]

    full_rows = [row for row in image_rows if row["subset"] == "full_l3sf"]
    annotated_image_rows = [row for row in image_rows if row["subset"] == "annotated_master"]
    full_identity_counts = Counter(row.get("canonical_identity") for row in full_rows)
    capture_pairs_by_identity: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in full_rows:
        if row.get("canonical_identity"):
            capture_pairs_by_identity[row["canonical_identity"]].add((row["capture_group"], row["instance"]))
    incomplete_full_identities = {
        identity: sorted(list(pairs))
        for identity, pairs in capture_pairs_by_identity.items()
        if pairs != {(group, instance) for group in (1, 2) for instance in range(1, 6)}
    }

    shapes = Counter(
        f"{row['width']}x{row['height']}x{row['channels']}"
        for row in image_rows
        if row["decode_status"] == "PASS"
    )
    formats = Counter(row["format"] for row in image_rows)
    pattern_counts = Counter(row.get("pattern") for row in annotated_image_rows)
    pattern_counts.pop(None, None)
    run_full_counts = Counter(row["run"] for row in full_rows)
    run_annotated_counts = Counter(row["run"] for row in annotated_image_rows)
    annotation_count_values = [row["annotation_count"] for row in annotation_file_rows]

    x_hist, x_edges = np.histogram(np.asarray(all_x_norm), bins=10, range=(0.0, 1.0))
    y_hist, y_edges = np.histogram(np.asarray(all_y_norm), bins=10, range=(0.0, 1.0))
    total_annotations = len(annotation_rows)
    total_oob = sum(row["out_of_bounds_count"] for row in annotation_file_rows)
    total_duplicate_records = sum(row["duplicate_coordinate_record_count"] for row in annotation_file_rows)
    duplicate_coordinate_affected_images = sum(
        row["duplicate_coordinate_record_count"] > 0 for row in annotation_file_rows
    )
    maximum_duplicate_records_in_one_image = max(
        (row["duplicate_coordinate_record_count"] for row in annotation_file_rows), default=0
    )
    total_edge16 = sum(row["within_16px_edge_count"] for row in annotation_file_rows)
    total_edge5pct = sum(row["within_5pct_edge_count"] for row in annotation_file_rows)

    dataset_inventory = {
        "experiment_id": "002-l3sf-pore-annotation-feasibility",
        "data_root_read_only": "${FINGERPRINT_DATASETS_ROOT}/L3_SF_V2/L3SF_V2",
        "observed_top_level_entries": sorted(path.name for path in DATA_ROOT.iterdir()),
        "all_image_count": len(image_rows),
        "image_formats": dict(formats),
        "decoded_image_shapes": dict(shapes),
        "decode_failure_count": sum(row["decode_status"] != "PASS" for row in image_rows),
        "full_l3sf": {
            "image_count": len(full_rows),
            "run_counts": dict(run_full_counts),
            "canonical_identity_count": len(full_identity_counts),
            "images_per_identity_distribution": dict(Counter(full_identity_counts.values())),
            "filename_semantics_observed": "R{run}/{numeric_identity}_{capture_group}_{instance}.png",
            "capture_group_values": sorted({row["capture_group"] for row in full_rows}),
            "instance_values": sorted({row["instance"] for row in full_rows}),
            "incomplete_identity_count": len(incomplete_full_identities),
        },
        "annotation_subset": {
            "image_count": len(annotated_image_rows),
            "annotation_file_count": len(annotation_file_rows),
            "run_counts": dict(run_annotated_counts),
            "pattern_counts": dict(pattern_counts),
            "canonical_annotated_identity_count": len(annotation_image_map),
            "total_annotation_count": total_annotations,
            "annotations_per_image": distribution(annotation_count_values),
            "image_annotation_correspondence": "Exact one-to-one pairing by R-directory and filename stem",
            "coordinate_fields": ["x", "y"],
            "coordinate_value_type": "integer",
            "coordinate_origin_and_axes": "Candidate zero-based x=column, y=row; range audit plus visual overlays required for semantic confirmation",
            "annotation_representation": "point_coordinate_only",
            "open_closed_pore_label_present": False,
            "additional_annotation_types_present": False,
        },
        "crosswalk_between_annotated_masters_and_full_numeric_identities": {
            "status": "NOT_EXPLICITLY_PRESENT_ON_DISK",
            "note": "The two branches use different filename schemes; no mapping file was found. Do not infer a crosswalk from numeric substrings alone.",
        },
    }

    integrity = {
        "experiment_id": "002-l3sf-pore-annotation-feasibility",
        "image_decode_failures": [row["relative_path"] for row in image_rows if row["decode_status"] != "PASS"],
        "full_filename_parse_failures": full_parse_failures,
        "annotation_image_name_parse_failures": ann_name_parse_failures,
        "orphan_annotation_files": orphan_tsv,
        "annotated_images_without_annotation_file": images_without_tsv,
        "malformed_annotation_files": malformed_tsv,
        "out_of_bounds_annotation_count": total_oob,
        "duplicate_coordinate_record_count": total_duplicate_records,
        "duplicate_coordinate_record_fraction": (
            total_duplicate_records / total_annotations if total_annotations else 0.0
        ),
        "duplicate_coordinate_affected_image_count": duplicate_coordinate_affected_images,
        "maximum_duplicate_coordinate_records_in_one_image": maximum_duplicate_records_in_one_image,
        "byte_identical_image_duplicate_group_count": len(duplicate_image_groups),
        "byte_identical_image_duplicate_groups": duplicate_image_groups,
        "byte_identical_annotation_file_duplicate_group_count": len(duplicate_annotation_file_groups),
        "byte_identical_annotation_file_duplicate_groups": duplicate_annotation_file_groups,
        "canonical_image_id_duplicate_count": len(image_rows)
        - len({(row["subset"], row.get("canonical_sample_id")) for row in image_rows}),
        "incomplete_full_identities": incomplete_full_identities,
        "integrity_systemic_problem": bool(
            any(row["decode_status"] != "PASS" for row in image_rows)
            or full_parse_failures
            or ann_name_parse_failures
            or orphan_tsv
            or images_without_tsv
            or malformed_tsv
            or total_oob
            or incomplete_full_identities
        ),
    }

    spatial = {
        "experiment_id": "002-l3sf-pore-annotation-feasibility",
        "annotation_count": total_annotations,
        "annotations_per_image": distribution(annotation_count_values),
        "density_per_100k_estimated_fingerprint_px": distribution(
            row["annotation_density_per_100k_estimated_fingerprint_px"] for row in per_image_spatial
        ),
        "nearest_neighbor_distance_px": distribution(all_nn),
        "normalized_x_histogram_10_bins": {"edges": x_edges.tolist(), "counts": x_hist.tolist()},
        "normalized_y_histogram_10_bins": {"edges": y_edges.tolist(), "counts": y_hist.tolist()},
        "within_16px_of_edge": {"count": total_edge16, "fraction": total_edge16 / total_annotations if total_annotations else None},
        "within_5pct_of_edge": {"count": total_edge5pct, "fraction": total_edge5pct / total_annotations if total_annotations else None},
        "exact_duplicate_coordinate_records": total_duplicate_records,
        "estimated_fingerprint_mask_method": "Otsu dark-ridge mask; 31x31 elliptical close; 9x9 dilation; retain components >=1% image area",
        "scientific_role": "Basic artifact/bias screen only; not an anatomical distribution model",
    }

    # Freeze the 50-image audit sample without decoding/displaying selected pixels.
    selected_images: list[dict[str, Any]] = []
    for run in [f"R{index}" for index in range(1, 6)]:
        for pattern, quota in PATTERN_QUOTAS_PER_RUN.items():
            candidates = [
                row
                for row in annotation_file_rows
                if row["run"] == run
                and row["pattern"] == pattern
                and row["canonical_image_id"] not in PRE_FREEZE_MAPPING_PILOT
            ]
            candidates.sort(key=lambda row: rank_hex("image", row["canonical_image_id"]))
            if len(candidates) < quota:
                raise RuntimeError(f"Insufficient audit candidates for {run}/{pattern}: {len(candidates)} < {quota}")
            for candidate in candidates[:quota]:
                selected_images.append(
                    {
                        **candidate,
                        "image_selection_rank_sha256": rank_hex("image", candidate["canonical_image_id"]),
                        "stratum_quota": quota,
                    }
                )

    selected_images.sort(key=lambda row: (row["run"], row["pattern"], row["image_selection_rank_sha256"]))
    if len(selected_images) != 50 or len({row["canonical_image_id"] for row in selected_images}) != 50:
        raise RuntimeError("Audit image selection must contain exactly 50 unique images")

    annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotation_rows:
        annotations_by_image[row["canonical_image_id"]].append(row)
    review_annotations: list[dict[str, Any]] = []
    review_images: list[dict[str, Any]] = []
    for audit_index, selected in enumerate(selected_images, start=1):
        candidates = annotations_by_image[selected["canonical_image_id"]]
        candidates.sort(
            key=lambda row: rank_hex(
                "annotation",
                row["canonical_image_id"],
                row["annotation_row_index"],
                row["x"],
                row["y"],
            )
        )
        chosen = candidates[: min(20, len(candidates))]
        review_image = {
            "audit_image_index": audit_index,
            "canonical_image_id": selected["canonical_image_id"],
            "run": selected["run"],
            "pattern": selected["pattern"],
            "image_relative_path": selected["image_relative_path"],
            "annotation_relative_path": selected["annotation_relative_path"],
            "image_sha256": selected["image_sha256"],
            "annotation_sha256": selected["annotation_sha256"],
            "annotation_count_in_image": selected["annotation_count"],
            "selected_annotation_count": len(chosen),
            "image_selection_rank_sha256": selected["image_selection_rank_sha256"],
        }
        review_images.append(review_image)
        for audit_annotation_index, row in enumerate(chosen, start=1):
            review_annotations.append(
                {
                    "audit_image_index": audit_index,
                    "audit_annotation_index": audit_annotation_index,
                    "canonical_image_id": row["canonical_image_id"],
                    "run": row["run"],
                    "pattern": row["pattern"],
                    "image_relative_path": row["image_relative_path"],
                    "annotation_relative_path": row["annotation_relative_path"],
                    "image_sha256": selected["image_sha256"],
                    "annotation_sha256": selected["annotation_sha256"],
                    "annotation_row_index": row["annotation_row_index"],
                    "x": row["x"],
                    "y": row["y"],
                    "annotation_selection_rank_sha256": rank_hex(
                        "annotation",
                        row["canonical_image_id"],
                        row["annotation_row_index"],
                        row["x"],
                        row["y"],
                    ),
                }
            )

    sample_manifest = {
        "experiment_id": "002-l3sf-pore-annotation-feasibility",
        "sampling_revision": 1,
        "audit_seed": AUDIT_SEED,
        "selection_rule": "Within each R1..R5, rank unseen annotated images by SHA256(seed|image|R/stem); select 4 right_loop, 3 whorl, 1 left_loop, 1 plain_arch, and 1 tented_arch. For each selected image, rank annotations by SHA256(seed|annotation|R/stem|row|x|y) and select the first min(20,n).",
        "image_count": len(review_images),
        "selected_annotation_count": len(review_annotations),
        "per_run_pattern_quotas": PATTERN_QUOTAS_PER_RUN,
        "pre_freeze_mapping_pilot_exclusions": sorted(PRE_FREEZE_MAPPING_PILOT),
        "pilot_exclusion_reason": "Displayed solely for on-disk structure/identity interpretation before audit freeze; excluded to preserve blindness, not because of image quality.",
        "selected_images_visually_viewed_before_manifest_freeze": False,
        "quality_based_selection_or_replacement": False,
        "selected_images": review_images,
    }

    annotations_path = INVENTORY_ROOT / "annotations.csv"

    write_csv(INVENTORY_ROOT / "dataset_images.csv", image_rows)
    write_csv(INVENTORY_ROOT / "annotation_files.csv", annotation_file_rows)
    write_csv(annotations_path, annotation_rows)
    write_csv(INVENTORY_ROOT / "spatial_per_image.csv", per_image_spatial)
    write_json(INVENTORY_ROOT / "dataset_inventory.json", dataset_inventory)
    write_json(INVENTORY_ROOT / "integrity_findings.json", integrity)
    write_json(INVENTORY_ROOT / "spatial_summary.json", spatial)
    write_json(
        INVENTORY_ROOT / "annotations.manifest.json",
        {
            "schema_version": 1,
            "artifact": "artifacts/experiment-002/inventory/annotations.csv",
            "tracked_in_git": False,
            "generator": "scripts/experiment_002_inventory_and_sample.py",
            "size_bytes": annotations_path.stat().st_size,
            "rows_including_header": len(annotation_rows) + 1,
            "data_rows": len(annotation_rows),
            "sha256": sha256_file(annotations_path),
        },
    )
    write_json(REVIEW_ROOT / "review_sample_manifest.json", sample_manifest)
    write_csv(REVIEW_ROOT / "review_sample_images.csv", review_images)
    write_csv(REVIEW_ROOT / "review_annotation_manifest.csv", review_annotations)

    print(f"Images inventoried: {len(image_rows)}")
    print(f"Annotated image/TSV pairs: {len(annotation_file_rows)}")
    print(f"Annotations: {total_annotations}")
    print(f"Audit sample: {len(review_images)} images, {len(review_annotations)} annotations")
    print(f"Integrity systemic problem: {integrity['integrity_systemic_problem']}")
    print(f"Wrote outputs under {OUT_ROOT}")


if __name__ == "__main__":
    main()
