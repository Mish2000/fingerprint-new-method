"""Decode frozen blind ratings and finalize Experiment 003 Gate 1."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from fingerprint_new_method.experiment003 import write_csv, write_json
from fingerprint_new_method.paths import PROJECT_ROOT

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "experiment-003"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    key_payload = json.loads((OUTPUT_ROOT / "crosswalk_review_blind_key.json").read_text(encoding="utf-8"))
    key_by_index = {int(row["audit_index"]): row for row in key_payload["rows"]}
    rating_rows = read_csv(OUTPUT_ROOT / "crosswalk_review_blind_ratings.csv")
    if len(rating_rows) != 50 or len({int(row["audit_index"]) for row in rating_rows}) != 50:
        raise RuntimeError("Blind review must contain exactly 50 unique ratings")

    decoded: list[dict[str, Any]] = []
    for rating in rating_rows:
        audit_index = int(rating["audit_index"])
        key = key_by_index[audit_index]
        choice = rating["reviewer_choice"]
        confidence = rating["confidence"]
        proposed_group = key["proposed_blind_group"]
        if choice == proposed_group:
            classification = "CLEAR" if confidence == "CLEAR" else "AMBIGUOUS"
            chosen_role = "PROPOSED"
        elif choice == "BOTH_UNCERTAIN":
            classification = "AMBIGUOUS"
            chosen_role = "BOTH_UNCERTAIN"
        elif choice == "NEITHER":
            classification = "NOT_MATCH"
            chosen_role = "NEITHER"
        elif choice in {"A", "B"}:
            classification = "NOT_MATCH"
            chosen_role = "ALTERNATIVE"
        else:
            raise ValueError(f"Invalid reviewer choice for audit {audit_index}: {choice}")
        decoded.append(
            {
                "audit_index": audit_index,
                "annotated_512_id": key["annotated_512_id"],
                "run": key["run"],
                "pattern": key["pattern"],
                "proposed_identity": key["proposed_identity"],
                "proposed_source": key["proposed_source"],
                "alternative_identity": key["alternative_identity"],
                "alternative_source": key["alternative_source"],
                "channels_agree_before_review": str(key["channels_agree"]).lower(),
                "proposed_blind_group": proposed_group,
                "reviewer_choice": choice,
                "reviewer_confidence": confidence,
                "chosen_role": chosen_role,
                "classification": classification,
                "reason": rating["reason"],
            }
        )
    decoded.sort(key=lambda row: row["audit_index"])
    fields = [
        "audit_index",
        "annotated_512_id",
        "run",
        "pattern",
        "proposed_identity",
        "proposed_source",
        "alternative_identity",
        "alternative_source",
        "channels_agree_before_review",
        "proposed_blind_group",
        "reviewer_choice",
        "reviewer_confidence",
        "chosen_role",
        "classification",
        "reason",
    ]
    write_csv(OUTPUT_ROOT / "crosswalk_review.csv", decoded, fields)

    counts = Counter(row["classification"] for row in decoded)
    clear_or_ambiguous = counts["CLEAR"] + counts["AMBIGUOUS"]
    review_summary = {
        "review_count": len(decoded),
        "classification_counts": dict(sorted(counts.items())),
        "clear_fraction": counts["CLEAR"] / len(decoded),
        "clear_or_ambiguous_fraction": clear_or_ambiguous / len(decoded),
        "review_90pct_threshold_pass": clear_or_ambiguous >= 45,
        "review_75pct_clear_threshold_pass": counts["CLEAR"] >= 38,
        "channels_agreed_in_frozen_review_before_visual_inspection": sum(
            row["channels_agree_before_review"] == "true" for row in decoded
        ),
        "protocol_contingency": key_payload["protocol_contingency"],
    }
    write_json(OUTPUT_ROOT / "crosswalk_review_summary.json", review_summary)

    crosswalk_summary_path = OUTPUT_ROOT / "crosswalk_summary.json"
    crosswalk_summary = json.loads(crosswalk_summary_path.read_text(encoding="utf-8"))
    crosswalk_summary["frozen_visual_review"] = review_summary
    crosswalk_summary["gate_1_pass"] = False
    crosswalk_summary["gate_1_status"] = "FAIL_NO_RELIABLE_CROSSWALK"
    crosswalk_summary["gate_1_failure_reasons"] = [
        "0 of 740 mappings met the preregistered STRONG definition; at least 703 were required.",
        "Only 4 of 740 local top-1 candidates agreed between the two independent channels.",
        "The frozen review cannot rescue an automatically ambiguous mapping and did not meet both visual thresholds.",
    ]
    crosswalk_summary["geometry_executed"] = False
    crosswalk_summary["geometry_stop_reason"] = "Gate 1 failed, so the preregistered stop rule prohibits annotation transfer."
    write_json(crosswalk_summary_path, crosswalk_summary)
    print(json.dumps(review_summary, indent=2))


if __name__ == "__main__":
    main()
