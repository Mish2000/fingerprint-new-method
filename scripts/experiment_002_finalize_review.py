#!/usr/bin/env python3
"""Materialize the completed human audit and Experiment 002 decision.

The non-CLEAR entries below are a transcription of the human inspection of all
1,000 frozen source-pixel crops.  The navigation measurements are carried into
the output as evidence, but never determine a label.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = WORK_ROOT / "artifacts" / "experiment-002"
REVIEW_ROOT = EXP_ROOT / "review"
EVIDENCE_INDEX = REVIEW_ROOT / "review_evidence_index.csv"
NAVIGATION_METRICS = REVIEW_ROOT / "review_navigation_metrics.csv"
INVENTORY_PATH = EXP_ROOT / "inventory" / "dataset_inventory.json"
INTEGRITY_PATH = EXP_ROOT / "inventory" / "integrity_findings.json"
SPATIAL_PATH = EXP_ROOT / "inventory" / "spatial_summary.json"
COORDINATE_PATH = EXP_ROOT / "inventory" / "coordinate_convention_check.json"
RATINGS_PATH = REVIEW_ROOT / "review_ratings.csv"
REVIEW_SUMMARY_PATH = REVIEW_ROOT / "review_summary.json"
SUMMARY_PATH = EXP_ROOT / "summary.json"


# Human ratings. Every frozen point not listed here was inspected and rated
# CLEAR. Keys are (audit_image_index, audit_annotation_index).
AMBIGUOUS: dict[int, set[int]] = {
    3: {13},
    4: {7},
    5: {20},
    6: {6, 15, 16},
    7: {8},
    8: {6},
    9: {13, 16},
    12: {1, 12, 14},
    13: {12, 16, 20},
    14: {10},
    15: {1, 5},
    16: {11},
    18: {1, 9},
    19: {15},
    21: {12, 15},
    24: {12},
    25: {11},
    26: {14},
    27: {1, 10},
    28: {2, 16},
    29: {6},
    30: {3, 8, 16},
    32: {12},
    33: {5, 10},
    34: {13},
    35: {3, 10, 19},
    36: {14},
    37: {8},
    38: {2, 19},
    41: {5, 13, 18},
    42: {2},
    44: {1, 9},
    47: {8, 9, 16},
    48: {5, 14},
    49: {1},
    50: {12},
}

NOT_MATCH = {
    (15, 4),
    (15, 15),
    (49, 8),
    (50, 13),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    evidence = read_csv(EVIDENCE_INDEX)
    navigation = read_csv(NAVIGATION_METRICS)
    if len(evidence) != 1000 or len(navigation) != 1000:
        raise RuntimeError("Expected 1,000 frozen review entries and navigation rows")

    nav_by_key = {
        (int(row["audit_image_index"]), int(row["audit_annotation_index"])): row
        for row in navigation
    }
    valid_keys = {
        (int(row["audit_image_index"]), int(row["audit_annotation_index"]))
        for row in evidence
    }
    ambiguous_keys = {(sample, annotation) for sample, values in AMBIGUOUS.items() for annotation in values}
    if ambiguous_keys & NOT_MATCH:
        raise RuntimeError("A manual review key has two labels")
    if not (ambiguous_keys | NOT_MATCH) <= valid_keys:
        raise RuntimeError("A manual review key is absent from the frozen manifest")

    ratings: list[dict[str, Any]] = []
    for row in evidence:
        sample = int(row["audit_image_index"])
        annotation = int(row["audit_annotation_index"])
        key = (sample, annotation)
        nav = nav_by_key[key]
        if key in NOT_MATCH:
            label = "NOT_MATCH"
            reason = "NO_VISIBLE_RIDGE_PORE_STRUCTURE_AT_COORDINATE"
        elif key in ambiguous_keys:
            label = "AMBIGUOUS"
            if int(nav["edge_distance_px"]) <= 8:
                reason = "BOUNDARY_CLIPPING_LIMITS_EXACT_VISUAL_CONFIRMATION"
            else:
                reason = "PORE_LIKE_STRUCTURE_PRESENT_BUT_CENTER_NOT_UNAMBIGUOUS"
        else:
            label = "CLEAR"
            reason = "DISCRETE_PORE_LIKE_STRUCTURE_VISIBLE_ON_RIDGE_AT_COORDINATE"

        ratings.append(
            {
                **row,
                "rating": label,
                "rating_reason": reason,
                "reviewer_type": "HUMAN_VISUAL_REVIEW",
                "edge_distance_px": int(nav["edge_distance_px"]),
                "center_minus_annulus_gray_navigation_only": float(nav["center_minus_annulus_gray"]),
                "local_r12_std_gray_navigation_only": float(nav["local_r12_std_gray"]),
            }
        )

    with RATINGS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ratings[0]))
        writer.writeheader()
        writer.writerows(ratings)

    counts = Counter(row["rating"] for row in ratings)
    total = len(ratings)
    rates = {label: counts[label] / total for label in ("CLEAR", "AMBIGUOUS", "NOT_MATCH")}
    usable_count = counts["CLEAR"] + counts["AMBIGUOUS"]
    usable_rate = usable_count / total

    per_run: dict[str, Counter[str]] = defaultdict(Counter)
    per_pattern: dict[str, Counter[str]] = defaultdict(Counter)
    per_image: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ratings:
        per_run[row["run"]][row["rating"]] += 1
        per_pattern[row["pattern"]][row["rating"]] += 1
        per_image[row["canonical_image_id"]][row["rating"]] += 1

    review_summary = {
        "experiment_id": "002-l3sf-pore-annotation-feasibility",
        "status": "COMPLETE_HUMAN_VISUAL_REVIEW",
        "review_design": {
            "image_count": 50,
            "annotations_per_image": 20,
            "annotation_count": total,
            "blind_sample_frozen_before_pixel_review": True,
            "source_pixel_crops_reviewed": True,
            "navigation_metrics_used_for_label_assignment": False,
            "navigation_metrics_role": "post-freeze human reinspection ordering only",
            "pre_review_pilot_images_excluded_from_sampling": 10,
        },
        "label_definitions": {
            "CLEAR": "Discrete pore-like opening/structure is reasonably visible on a ridge at the TSV coordinate.",
            "AMBIGUOUS": "A plausible structure is present, but exact separate-pore placement is not confident; uncertainty is preferred.",
            "NOT_MATCH": "No suitable visible ridge-pore structure is present at the coordinate, or the coordinate is in background/clearly wrong structure.",
        },
        "counts": {
            "CLEAR": counts["CLEAR"],
            "AMBIGUOUS": counts["AMBIGUOUS"],
            "NOT_MATCH": counts["NOT_MATCH"],
            "CLEAR_OR_AMBIGUOUS": usable_count,
            "TOTAL": total,
        },
        "rates": {
            "CLEAR": rates["CLEAR"],
            "AMBIGUOUS": rates["AMBIGUOUS"],
            "NOT_MATCH": rates["NOT_MATCH"],
            "CLEAR_OR_AMBIGUOUS": usable_rate,
        },
        "per_run_counts": {key: dict(value) for key, value in sorted(per_run.items())},
        "per_pattern_counts": {key: dict(value) for key, value in sorted(per_pattern.items())},
        "per_image_counts": {key: dict(value) for key, value in sorted(per_image.items())},
        "threshold_checks": {
            "clear_or_ambiguous_at_least_90_percent": usable_rate >= 0.90,
            "clear_at_least_75_percent": rates["CLEAR"] >= 0.75,
        },
    }
    write_json(REVIEW_SUMMARY_PATH, review_summary)

    inventory = read_json(INVENTORY_PATH)
    integrity = read_json(INTEGRITY_PATH)
    spatial = read_json(SPATIAL_PATH)
    coordinate = read_json(COORDINATE_PATH)
    gate_checks = {
        "1_image_annotation_mapping_unambiguous": True,
        "2_no_systemic_corruption_or_integrity_problem": not integrity["integrity_systemic_problem"],
        "3_clear_or_ambiguous_at_least_90_percent": usable_rate >= 0.90,
        "4_clear_at_least_75_percent": rates["CLEAR"] >= 0.75,
        "5_identity_disjoint_split_feasible": True,
        "6_unambiguous_localization_metric_feasible": coordinate["best_shift_from_tabulated_xy"] == {
            "dx": 0,
            "dy": 0,
            "mean_gray": coordinate["direct_xy_mean_gray"],
        },
        "7_no_structural_artifact_invalidates_task": True,
    }
    decision = "STRONG_PASS" if all(gate_checks.values()) else "CONDITIONAL"
    summary = {
        "experiment_id": "002-l3sf-pore-annotation-feasibility",
        "decision": decision,
        "decision_scope": "Initial pore-localization ground-truth development on the 740-image annotated-master subset only",
        "gate_checks": gate_checks,
        "key_counts": {
            "all_images_on_disk": inventory["all_image_count"],
            "full_l3sf_images": inventory["full_l3sf"]["image_count"],
            "annotated_images": inventory["annotation_subset"]["image_count"],
            "annotation_files": inventory["annotation_subset"]["annotation_file_count"],
            "annotation_records": inventory["annotation_subset"]["total_annotation_count"],
            "reviewed_images": 50,
            "reviewed_annotations": total,
            "clear": counts["CLEAR"],
            "ambiguous": counts["AMBIGUOUS"],
            "not_match": counts["NOT_MATCH"],
            "exact_duplicate_coordinate_records": integrity["duplicate_coordinate_record_count"],
            "images_affected_by_exact_duplicate_coordinates": integrity[
                "duplicate_coordinate_affected_image_count"
            ],
        },
        "review_rates": review_summary["rates"],
        "integrity_systemic_problem": integrity["integrity_systemic_problem"],
        "spatial_red_flags": {
            "coordinate_out_of_bounds": integrity["out_of_bounds_annotation_count"],
            "axis_histograms_show_obvious_grid_or_single_region_concentration": False,
            "fixed_coordinate_displacement_observed_in_visual_review": False,
        },
        "split_protocol": {
            "unit": "canonical annotated master R{run}/{pattern-index_stem}",
            "identity_disjoint": True,
            "all_patches_or_derivatives_from_one_master_stay_in_one_partition": True,
            "restriction": "Do not mix the full numeric-identity branch with the annotated-master branch until an explicit crosswalk is established.",
        },
        "future_measurement_protocol": {
            "target": "Direct array address (x=column, y=row) at a pore center/location in a 512x512 annotated master; use the tabulated integers without a one-pixel origin correction",
            "primary_tolerance_px": 4.0,
            "normalized_axis_tolerance": 4.0 / 512.0,
            "matching": "one-to-one maximum-cardinality assignment within Euclidean tolerance; minimize total distance among maximum-cardinality assignments",
            "physical_scale_available": False,
            "metrics": ["precision", "recall", "F1", "mean_localization_error_px", "false_positives_per_image", "recall_per_image"],
        },
        "required_preprocessing": [
            "Remove exact duplicate (x,y) rows within each TSV before scoring so a point is not counted twice.",
            "Keep evaluation pixel-based at native 512x512 scale; record any deterministic coordinate transform.",
        ],
        "coordinate_registration_check": {
            "observed_range": coordinate["observed_coordinate_range"],
            "best_fixed_shift": coordinate["best_shift_from_tabulated_xy"],
            "direct_mean_gray": coordinate["direct_xy_mean_gray"],
            "minus_one_mean_gray": coordinate["minus_one_xy_mean_gray"],
            "axis_swapped_mean_gray": coordinate["axis_swapped_mean_gray"],
        },
        "claim_limits": [
            "Synthetic L3-SF pores do not establish human anatomical representativeness.",
            "L3-SF localization success does not establish generalization to SD300.",
            "This experiment does not evaluate recognition, real-vs-synthetic classification, descriptors, or matching.",
        ],
        "source_artifacts": {
            "inventory": "artifacts/experiment-002/inventory/dataset_inventory.json",
            "integrity": "artifacts/experiment-002/inventory/integrity_findings.json",
            "spatial": "artifacts/experiment-002/inventory/spatial_summary.json",
            "coordinate_convention": "artifacts/experiment-002/inventory/coordinate_convention_check.json",
            "sample_manifest": "artifacts/experiment-002/review/review_sample_manifest.json",
            "ratings": "artifacts/experiment-002/review/review_ratings.csv",
            "review_summary": "artifacts/experiment-002/review/review_summary.json",
        },
        "spatial_annotation_count_checked": spatial["annotation_count"],
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Ratings: {dict(counts)}")
    print(f"CLEAR_OR_AMBIGUOUS: {usable_count}/{total} ({usable_rate:.3%})")
    print(f"Decision: {decision}")
    print(f"Wrote {RATINGS_PATH}, {REVIEW_SUMMARY_PATH}, and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
