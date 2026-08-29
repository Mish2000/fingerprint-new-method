#!/usr/bin/env python3
"""Freeze Experiment 001 ratings into auditable measurement and summary files."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts" / "experiment-001"
SELECTION_PATH = EXP / "selection" / "selection_manifest.json"
RATINGS_PATH = EXP / "measurements" / "ratings.json"
CROSS_RES_PATH = EXP / "analysis" / "cross_resolution_metrics.json"
OUT_CSV = EXP / "measurements" / "measurements.csv"
OUT_SUMMARY = EXP / "measurements" / "summary.json"

RATING_FIELDS = (
    "plain_2000_observability",
    "roll_2000_observability",
    "plain_cross_resolution_1000_2000",
    "roll_cross_resolution_1000_2000",
    "cross_impression_repeatability",
)
ALLOWED = {"CLEAR", "AMBIGUOUS", "NONE"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_classification(rating: dict[str, Any]) -> str:
    if all(rating[field] == "CLEAR" for field in RATING_FIELDS):
        return "LEVEL3_USABLE"
    if (
        rating["plain_cross_resolution_1000_2000"] == "CLEAR"
        and rating["roll_cross_resolution_1000_2000"] == "CLEAR"
        and rating["cross_impression_repeatability"] != "CLEAR"
    ):
        return "SCAN_REPEATABLE_BUT_ANATOMICALLY_UNVALIDATED"
    return "NOT_LEVEL3_USABLE"


def main() -> None:
    selection = load_json(SELECTION_PATH)
    ratings = load_json(RATINGS_PATH)
    cross_res = load_json(CROSS_RES_PATH)

    selected = selection["selected"]
    if len(selected) != 20 or len(ratings) != 20:
        raise RuntimeError("Experiment 001 must contain exactly 20 selected fingers and 20 ratings")
    if len({row["subject_id"] for row in selected}) != 20:
        raise RuntimeError("Selected subject IDs are not unique")
    position_counts = Counter(row["anatomical_position"] for row in selected)
    if position_counts != Counter({position: 2 for position in range(1, 11)}):
        raise RuntimeError(f"Expected exactly two samples per position, got {position_counts}")

    by_index = {row["sample_index"]: row for row in ratings}
    if set(by_index) != set(range(1, 21)):
        raise RuntimeError("Rating sample indices must be exactly 1..20")
    for rating in ratings:
        for field in RATING_FIELDS:
            if rating.get(field) not in ALLOWED:
                raise RuntimeError(f"Invalid {field} for S{rating['sample_index']:02}: {rating.get(field)}")
        evidence_path = ROOT / rating["evidence_reference"]
        if not evidence_path.is_file():
            raise RuntimeError(f"Missing evidence reference: {evidence_path}")

    metric_by_key = {(row["sample_index"], row["impression"]): row for row in cross_res}
    if len(metric_by_key) != 40:
        raise RuntimeError("Expected 40 cross-resolution metric records")

    rows: list[dict[str, Any]] = []
    for sample in selected:
        index = sample["sample_index"]
        rating = by_index[index]
        classification = final_classification(rating)
        row: dict[str, Any] = {
            "sample_index": index,
            "subject_id": sample["subject_id"],
            "anatomical_position": sample["anatomical_position"],
            "finger_name": sample["finger_name"],
            "plain_2000_observability": rating["plain_2000_observability"],
            "roll_2000_observability": rating["roll_2000_observability"],
            "plain_cross_resolution_1000_2000": rating["plain_cross_resolution_1000_2000"],
            "roll_cross_resolution_1000_2000": rating["roll_cross_resolution_1000_2000"],
            "cross_impression_repeatability": rating["cross_impression_repeatability"],
            "final_classification": classification,
            "level3_usable": classification == "LEVEL3_USABLE",
            "reason": rating["reason"],
            "evidence_reference": rating["evidence_reference"],
            "control_500_role": "DESCRIPTIVE_ONLY_NOT_USED_FOR_DECISION",
        }

        for source in sample["sources"]:
            prefix = f"{source['impression']}_{source['ppi']}"
            if source["official_sha256"] != source["actual_sha256"]:
                raise RuntimeError(f"Source hash mismatch in selected sample S{index:02}: {source['source_id']}")
            row[f"{prefix}_source_id"] = source["source_id"]
            row[f"{prefix}_sha256"] = source["actual_sha256"]

        for impression in ("plain", "roll"):
            metric = metric_by_key[(index, impression)]
            row[f"{impression}_1000_2000_pearson_all_pixels"] = round(metric["pearson_all_pixels"], 6)
            row[f"{impression}_1000_2000_pearson_dark_union"] = round(metric["pearson_dark_union"], 6)
            row[f"{impression}_1000_2000_mae_all_pixels"] = round(metric["mae_all_pixels"], 6)
            row[f"{impression}_500_1000_control_pearson_all_pixels"] = round(
                metric["control_500_1000_pearson_all_pixels"], 6
            )

        sample_key = f"S{index:02}_{sample['subject_id']}_P{sample['anatomical_position']:02}"
        evidence_manifest_path = EXP / "evidence-pixels" / sample_key / "evidence_manifest.json"
        evidence_manifest = load_json(evidence_manifest_path)
        alignment = evidence_manifest["navigation_alignment"]
        row["navigation_alignment_scientific_status"] = alignment["scientific_status"]
        row["navigation_alignment_source_ppi"] = alignment[
            "chosen_source_ppi_downsampled_to_500_for_navigation"
        ]
        row["navigation_alignment_geometric_inliers"] = alignment["geometric_inlier_count"]
        row["navigation_patch_count"] = len(evidence_manifest["patches"])
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    class_counts = Counter(row["final_classification"] for row in rows)
    usable = class_counts["LEVEL3_USABLE"]
    decision = "STRONG_PASS" if usable >= 14 else "CONDITIONAL" if usable >= 6 else "FAIL"
    rating_counts = {
        field: dict(Counter(row[field] for row in rows))
        for field in RATING_FIELDS
    }
    dark_union_values = [
        metric["pearson_dark_union"] for metric in cross_res
    ]
    summary = {
        "experiment_id": selection["experiment_id"],
        "selection_revision": selection["selection_revision"],
        "sampling_seed": selection["sampling_seed"],
        "selected_finger_count": len(rows),
        "unique_subject_count": len({row["subject_id"] for row in rows}),
        "per_position_counts": {str(k): position_counts[k] for k in sorted(position_counts)},
        "selected_source_integrity": "PASS_ALL_120_FILES_MATCH_OFFICIAL_SHA256",
        "level3_usable_count": usable,
        "level3_usable_fraction": usable / len(rows),
        "decision": decision,
        "decision_rule": "STRONG_PASS=14..20; CONDITIONAL=6..13; FAIL=0..5",
        "final_classification_counts": dict(class_counts),
        "rating_counts": rating_counts,
        "cross_resolution_1000_2000_dark_union_pearson_descriptive": {
            "n_impressions": len(dark_union_values),
            "minimum": min(dark_union_values),
            "median": statistics.median(dark_union_values),
            "maximum": max(dark_union_values),
            "scientific_role": "DESCRIPTIVE_SUPPORT_ONLY_NOT_A_CLASSIFICATION_THRESHOLD",
        },
        "objective_exclusions_and_replacements": selection.get("exclusions_and_replacements", []),
        "ratings_sha256": sha256(RATINGS_PATH),
        "selection_manifest_sha256": sha256(SELECTION_PATH),
        "measurement_csv_sha256": sha256(OUT_CSV),
        "review_policy": "Conservative visual review of original source pixels; when in doubt AMBIGUOUS; detector and alignment outputs used for navigation only.",
        "pixel_evidence_git_policy": "LOCAL_ONLY_EXCLUDED_FROM_GIT",
    }
    with OUT_SUMMARY.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Decision: {decision}; usable: {usable}/20")


if __name__ == "__main__":
    main()
