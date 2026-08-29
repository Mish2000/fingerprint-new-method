"""Finalize the stopped Experiment 003 result and compact summary artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from fingerprint_new_method.experiment003 import write_csv, write_json
from fingerprint_new_method.paths import PROJECT_ROOT

EXPERIMENT_ID = "003-l3sf-annotated-final-crosswalk"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "experiment-003"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "p05": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def channel_diagnostics(rows: list[dict[str, str]], channel: str) -> dict[str, Any]:
    support_pass = [row[f"{channel}_support_pass"] == "true" for row in rows]
    margin_pass = [row[f"{channel}_margin_pass"] == "true" for row in rows]
    loo_pass = [row[f"{channel}_winner_leave_one_out_stable"] == "true" for row in rows]
    all_pass = [all(values) for values in zip(support_pass, margin_pass, loo_pass, strict=True)]
    return {
        "support_pass_count": sum(support_pass),
        "margin_pass_count": sum(margin_pass),
        "winner_leave_one_out_stable_count": sum(loo_pass),
        "all_three_channel_conditions_pass_count": sum(all_pass),
        "robust_separation": distribution([float(row[f"{channel}_robust_separation"]) for row in rows]),
        "support_impression_count_distribution": dict(
            sorted(Counter(int(row[f"{channel}_support_impression_count"]) for row in rows).items())
        ),
        "global_position_drop_change_count_distribution": dict(
            sorted(Counter(int(row[f"{channel}_global_position_drop_change_count"]) for row in rows).items())
        ),
        "distinct_top1_identities_per_run": {
            run: len({int(row[f"{channel}_top1_identity"]) for row in rows if row["run"] == run})
            for run in (f"R{index}" for index in range(1, 6))
        },
    }


def main() -> None:
    crosswalk_summary = json.loads((OUTPUT_ROOT / "crosswalk_summary.json").read_text(encoding="utf-8"))
    if crosswalk_summary.get("gate_1_pass") is not False:
        raise RuntimeError("This finalizer is for the preregistered Gate 1 stop outcome")
    crosswalk_rows = read_csv(OUTPUT_ROOT / "crosswalk.csv")
    review_rows = read_csv(OUTPUT_ROOT / "crosswalk_review.csv")
    if len(crosswalk_rows) != 740 or len(review_rows) != 50:
        raise RuntimeError("Unexpected crosswalk or frozen-review cardinality")

    registration_path = OUTPUT_ROOT / "registration_summary.csv"
    pore_review_path = OUTPUT_ROOT / "pore_transfer_review.csv"
    write_csv(
        registration_path,
        [],
        [
            "pair_id",
            "annotated_512_id",
            "final_320_sample_id",
            "registration_status",
            "not_executed_reason",
        ],
    )
    write_csv(
        pore_review_path,
        [],
        [
            "pair_id",
            "annotation_x",
            "annotation_y",
            "predicted_final_x",
            "predicted_final_y",
            "classification",
            "not_executed_reason",
        ],
    )

    channel_details = {
        "K_rootsift_affine": channel_diagnostics(crosswalk_rows, "k"),
        "O_ridge_orientation_affine": channel_diagnostics(crosswalk_rows, "o"),
    }
    crosswalk_summary["channel_diagnostics"] = channel_details
    write_json(OUTPUT_ROOT / "crosswalk_summary.json", crosswalk_summary)

    artifact_names = [
        "semantic_audit.json",
        "crosswalk.csv",
        "crosswalk_summary.json",
        "crosswalk_review.csv",
        "crosswalk_review_summary.json",
        "score_matrices.manifest.json",
        "registration_summary.csv",
        "pore_transfer_review.csv",
    ]
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_STOPPED_AT_GATE_1",
        "decision": "NO_RELIABLE_CROSSWALK",
        "supported_hypothesis": "H0",
        "terminology": {"annotated_branch": "annotated_512", "final_branch": "final_320"},
        "question_a": {
            "answer": "EXACT_STAGE_NOT_PROVED",
            "supported_role": "A pore-bearing, pore-annotated representation at or after pore insertion and before or outside the distributed final_320 branch.",
            "plausible_but_unproved_role": "The post-pore/post-scratch L3 Master Fingerprint described in the local method text, before acquisition simulation.",
            "reason": "The local text defines that pipeline stage, and each annotated_512 has pore coordinates, but no file-level provenance links the distributed 512 images to a named generation stage.",
        },
        "question_b": {
            "answer": "NO_RELIABLE_IDENTITY_CROSSWALK",
            "strong_correspondence_count": 0,
            "required_strong_correspondence_count": 703,
            "channel_top1_agreement_count": 4,
            "channel_top1_agreement_fraction": 4 / 740,
            "frozen_review_clear_count": 28,
            "frozen_review_clear_or_ambiguous_count": 40,
            "frozen_review_count": 50,
        },
        "question_c": {
            "answer": "NOT_EVALUATED_BY_STOP_RULE",
            "registration_pair_count": 0,
            "pore_transfer_point_count": 0,
            "reason": "Gate 1 failed; annotations must not be transferred without an identity crosswalk.",
        },
        "gate_1": {
            "pass": False,
            "automatic_strong_fraction": 0.0,
            "visual_clear_fraction": 28 / 50,
            "visual_clear_or_ambiguous_fraction": 40 / 50,
            "failure_reasons": crosswalk_summary["gate_1_failure_reasons"],
        },
        "gate_2": {"status": "NOT_RUN", "reason": crosswalk_summary["geometry_stop_reason"]},
        "methods": {
            "candidate_groups_scored": 740 * 148,
            "impression_pairs_scored_per_channel": 740 * 1480,
            "cross_run_search_performed": False,
            "pore_annotations_used_for_crosswalk": False,
            "one_to_one_assignment_used_as_proof": False,
            "channel_diagnostics": channel_details,
        },
        "interpretation_limit": "This decision means the required evidence is insufficient; it does not prove that no latent relationship exists between the distributed branches.",
        "source_dataset_writes": 0,
        "pixel_evidence_tracked_in_git": False,
        "artifacts": {
            name: {"size_bytes": (OUTPUT_ROOT / name).stat().st_size, "sha256": sha256_file(OUTPUT_ROOT / name)}
            for name in artifact_names
        },
    }
    write_json(OUTPUT_ROOT / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "summary": str(OUTPUT_ROOT / 'summary.json')}, indent=2))


if __name__ == "__main__":
    main()
