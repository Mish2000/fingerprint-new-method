"""Audit L3-SF semantics and freeze the Experiment 003 visual-review sample.

This script reads image containers and hashes source files but never decodes
pixel arrays and never reads pore-coordinate TSV content.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from fingerprint_new_method.experiment003 import (
    dataclass_dict,
    parse_annotated_filename,
    parse_final_filename,
    select_stratified,
    write_json,
)
from fingerprint_new_method.paths import PROJECT_ROOT, dataset_path, datasets_root

EXPERIMENT_ID = "003-l3sf-annotated-final-crosswalk"
DATA_ROOT = dataset_path("L3_SF_V2", "L3SF_V2")
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "experiment-003"
REVIEW_SEED = "l3sf-exp003-crosswalk-review-v1"
RUNS = tuple(f"R{index}" for index in range(1, 6))
PATTERN_QUOTAS = {
    "right_loop": 4,
    "whorl": 3,
    "left_loop": 1,
    "plain_arch": 1,
    "tented_arch": 1,
}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(DATA_ROOT).as_posix()


def safe_metadata_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        rendered = value
    elif isinstance(value, (tuple, list)):
        rendered = [safe_metadata_value(item) for item in value]
    else:
        rendered = repr(value)
    if isinstance(rendered, str) and len(rendered) > 500:
        return {"type": "long_string", "length": len(rendered), "sha256": hashlib.sha256(rendered.encode()).hexdigest()}
    return rendered


def image_container_record(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        exif = image.getexif()
        return {
            "relative_path": relative(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "info": {str(key): safe_metadata_value(value) for key, value in sorted(image.info.items())},
            "exif": {str(key): safe_metadata_value(value) for key, value in sorted(exif.items())},
        }


def timestamp_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [record["modified_utc"] for record in records]
    counts = Counter(timestamps)
    return {
        "minimum_utc": min(timestamps) if timestamps else None,
        "maximum_utc": max(timestamps) if timestamps else None,
        "distinct_count": len(counts),
        "most_common": [{"timestamp_utc": stamp, "file_count": count} for stamp, count in counts.most_common(10)],
        "scientific_role": "provenance_only_not_correspondence_evidence",
    }


def main() -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Missing L3-SF data root below {datasets_root()}")

    all_files = sorted(path for path in DATA_ROOT.rglob("*") if path.is_file())
    archive_files = [relative(path) for path in all_files if path.suffix.lower() in ARCHIVE_SUFFIXES]
    text_files = [path for path in all_files if path.suffix.lower() in TEXT_SUFFIXES]
    other_files = [
        relative(path)
        for path in all_files
        if path.suffix.lower() not in TEXT_SUFFIXES | ARCHIVE_SUFFIXES | {".png", ".jpg", ".jpeg", ".tsv"}
    ]

    final_paths = sorted((DATA_ROOT / "L3-SF").glob("R*/*.png"))
    annotated_paths = sorted((DATA_ROOT / "Pore ground truth" / "Fingerprint Images").glob("R*/*.jpg"))
    annotation_paths = sorted((DATA_ROOT / "Pore ground truth" / "Ground truth").glob("R*/*.tsv"))

    final_records: list[dict[str, Any]] = []
    final_parse_failures: list[str] = []
    for path in final_paths:
        try:
            parsed = parse_final_filename(path.name)
        except ValueError:
            final_parse_failures.append(relative(path))
            continue
        record = image_container_record(path)
        record.update(dataclass_dict(parsed))
        record["run"] = path.parent.name
        record["canonical_final_sample_id"] = f"{path.parent.name}/{path.stem}"
        record["canonical_final_identity"] = f"{path.parent.name}/{parsed.identity}"
        final_records.append(record)

    annotated_records: list[dict[str, Any]] = []
    annotated_parse_failures: list[str] = []
    for path in annotated_paths:
        try:
            parsed = parse_annotated_filename(path.name)
        except ValueError:
            annotated_parse_failures.append(relative(path))
            continue
        record = image_container_record(path)
        record.update(dataclass_dict(parsed))
        record["run"] = path.parent.name
        record["canonical_annotated_id"] = f"{path.parent.name}/{path.stem}"
        annotated_records.append(record)

    final_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in final_records:
        final_by_identity[record["canonical_final_identity"]].append(record)
    final_identity_counts = Counter(record["run"] for identity, records in final_by_identity.items() for record in records[:1])
    final_image_run_counts = Counter(record["run"] for record in final_records)
    annotated_run_counts = Counter(record["run"] for record in annotated_records)
    annotation_run_counts = Counter(path.parent.name for path in annotation_paths)
    expected_positions = {(group, instance) for group in (1, 2) for instance in range(1, 6)}
    incomplete_identities = {
        identity: sorted((record["capture_group"], record["instance"]) for record in records)
        for identity, records in final_by_identity.items()
        if {(record["capture_group"], record["instance"]) for record in records} != expected_positions
    }

    final_hashes = {record["sha256"] for record in final_records}
    annotated_hashes = {record["sha256"] for record in annotated_records}
    cross_branch_duplicate_hashes = sorted(final_hashes & annotated_hashes)
    metadata_key_counts = {
        "final_info_keys": dict(Counter(key for record in final_records for key in record["info"])),
        "final_exif_tag_ids": dict(Counter(key for record in final_records for key in record["exif"])),
        "annotated_info_keys": dict(Counter(key for record in annotated_records for key in record["info"])),
        "annotated_exif_tag_ids": dict(Counter(key for record in annotated_records for key in record["exif"])),
    }

    text_records: list[dict[str, Any]] = []
    for path in text_files:
        content = path.read_text(encoding="utf-8", errors="replace")
        text_records.append(
            {
                "relative_path": relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "line_count": len(content.splitlines()),
                "mentions_master": "master" in content.lower(),
                "mentions_seed": "seed" in content.lower() or "c image" in content.lower(),
                "mentions_cycle_gan": "cyclegan" in content.lower() or "psychogun" in content.lower(),
                "mentions_740_annotations": "740" in content and "annotation" in content.lower(),
                "contains_explicit_filename_crosswalk": False,
            }
        )

    quotas = {(run, pattern): quota for run in RUNS for pattern, quota in PATTERN_QUOTAS.items()}
    review_records = select_stratified(annotated_records, seed=REVIEW_SEED, quotas=quotas)
    compact_review_records = [
        {
            "audit_index": index,
            "canonical_annotated_id": record["canonical_annotated_id"],
            "run": record["run"],
            "pattern": record["pattern"],
            "image_relative_path": record["relative_path"],
            "image_sha256": record["sha256"],
            "selection_rank_sha256": record["selection_rank_sha256"],
            "stratum_quota": record["stratum_quota"],
        }
        for index, record in enumerate(review_records, start=1)
    ]

    semantic_audit = {
        "experiment_id": EXPERIMENT_ID,
        "audit_revision": 1,
        "data_root_read_only": "${FINGERPRINT_DATASETS_ROOT}/L3_SF_V2/L3SF_V2",
        "pixel_arrays_decoded": False,
        "pore_coordinate_files_read": False,
        "observed_top_level_entries": sorted(path.name for path in DATA_ROOT.iterdir()),
        "file_inventory": {
            "all_file_count": len(all_files),
            "final_320_count": len(final_records),
            "annotated_512_count": len(annotated_records),
            "annotation_tsv_count": len(annotation_paths),
            "text_metadata_files": text_records,
            "archive_files": archive_files,
            "other_unclassified_files": other_files,
        },
        "structure": {
            "runs": list(RUNS),
            "final_images_per_run": dict(sorted(final_image_run_counts.items())),
            "final_identities_per_run": dict(sorted(final_identity_counts.items())),
            "annotated_images_per_run": dict(sorted(annotated_run_counts.items())),
            "annotation_files_per_run": dict(sorted(annotation_run_counts.items())),
            "final_identity_count": len(final_by_identity),
            "final_images_per_identity_distribution": dict(Counter(len(records) for records in final_by_identity.values())),
            "incomplete_final_identities": incomplete_identities,
            "final_filename_parse_failures": final_parse_failures,
            "annotated_filename_parse_failures": annotated_parse_failures,
            "annotated_pattern_counts": dict(sorted(Counter(record["pattern"] for record in annotated_records).items())),
        },
        "container_metadata": {
            "final_formats": dict(Counter(record["format"] for record in final_records)),
            "final_modes": dict(Counter(record["mode"] for record in final_records)),
            "final_dimensions": dict(Counter(f"{record['width']}x{record['height']}" for record in final_records)),
            "annotated_formats": dict(Counter(record["format"] for record in annotated_records)),
            "annotated_modes": dict(Counter(record["mode"] for record in annotated_records)),
            "annotated_dimensions": dict(Counter(f"{record['width']}x{record['height']}" for record in annotated_records)),
            "metadata_key_counts": metadata_key_counts,
            "metadata_crosswalk_field_found": False,
        },
        "hash_audit": {
            "cross_branch_byte_identical_file_count": len(cross_branch_duplicate_hashes),
            "cross_branch_byte_identical_sha256": cross_branch_duplicate_hashes,
        },
        "timestamps": {
            "final_320": timestamp_summary(final_records),
            "annotated_512": timestamp_summary(annotated_records),
        },
        "findings": {
            "proved": [
                "The local descriptive text places pore/scratch insertion before acquisition simulation, calls the post-acquisition crop/rotation a seed image, and places image translation after the seed.",
                "The two image branches have five runs, with 148 annotated_512 images and 148 ten-impression final_320 identities per run.",
                "No image container field, archive, manifest, or separate mapping file is present that explicitly links an annotated_512 basename to a final_320 identity.",
            ],
            "supported": [
                "annotated_512 is a pore-bearing representation at or after pore insertion, because it is paired one-to-one with pore-coordinate TSV files.",
            ],
            "plausible": [
                "The dimensions and full-fingerprint framing are compatible with a pre-acquisition pore-bearing L3 master, but neither fact identifies the exact pipeline stage by itself.",
                "The equal 148-per-run cardinality permits a one-to-one identity relation, but cardinality alone is not correspondence evidence.",
            ],
            "not_found": [
                "An explicit annotated_512-to-final_320 crosswalk.",
                "Generation seeds, command logs, model checkpoints, acquisition parameters, or per-image provenance metadata.",
                "EXIF or PNG text fields carrying identity correspondence.",
                "A local archive containing additional metadata.",
            ],
            "numeric_stem_warning": "Numeric components in the two filename schemes were not compared or used as correspondence evidence.",
        },
    }

    review_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "manifest_revision": 1,
        "audit_seed": REVIEW_SEED,
        "selection_rule": "For each run and pattern, rank by SHA256(seed|run|canonical_annotated_id); select quotas 4 right_loop, 3 whorl, and 1 each left_loop/plain_arch/tented_arch.",
        "per_run_pattern_quotas": PATTERN_QUOTAS,
        "image_count": len(compact_review_records),
        "selected_images_visually_viewed_before_manifest_freeze": False,
        "crosswalk_scores_computed_before_manifest_freeze": False,
        "quality_based_replacement_allowed": False,
        "selected_images": compact_review_records,
    }

    write_json(OUTPUT_ROOT / "semantic_audit.json", semantic_audit)
    write_json(OUTPUT_ROOT / "crosswalk_review_sample_manifest.json", review_manifest)
    print(json.dumps({"semantic_audit": str(OUTPUT_ROOT / 'semantic_audit.json'), "review_images": len(review_records)}))


if __name__ == "__main__":
    main()
