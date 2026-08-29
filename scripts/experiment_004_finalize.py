"""Finalize Experiment 004 at its preregistered stopping condition."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fingerprint_new_method.experiment004 import (
    ARTIFACT_ROOT,
    EXPERIMENT_ID,
    PROJECT_ROOT,
    sha256_file,
    write_csv,
    write_json,
)

RESULTS_PATH = PROJECT_ROOT / "docs" / "experiments" / "004-pore-localization-and-sd300-transfer-results.md"
RATING_FIELDS = (
    "plain_2000_observability",
    "roll_2000_observability",
    "plain_cross_resolution_1000_2000",
    "roll_cross_resolution_1000_2000",
    "cross_impression_repeatability",
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _classification(rating: dict[str, Any]) -> str:
    if all(rating[field] == "CLEAR" for field in RATING_FIELDS):
        return "LEVEL3_USABLE"
    if (
        rating["plain_cross_resolution_1000_2000"] == "CLEAR"
        and rating["roll_cross_resolution_1000_2000"] == "CLEAR"
        and rating["cross_impression_repeatability"] != "CLEAR"
    ):
        return "SCAN_REPEATABLE_BUT_ANATOMICALLY_UNVALIDATED"
    return "NOT_LEVEL3_USABLE"


def _format_float(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _write_not_run_sd300() -> None:
    write_json(
        ARTIFACT_ROOT / "sd300_pair_manifest.json",
        {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "NOT_RUN_GATE_A_FAIL", "records": []},
    )
    write_csv(
        ARTIFACT_ROOT / "sd300_registration_summary.csv",
        [],
        ["record_id", "ppi", "pair_type", "status", "reason"],
    )
    write_csv(
        ARTIFACT_ROOT / "sd300_repeatability.csv",
        [],
        ["record_id", "sample_index", "ppi", "seed", "mated_repeatability", "non_mated_repeatability", "delta"],
    )
    write_json(
        ARTIFACT_ROOT / "sd300_transfer_summary.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "NOT_RUN_GATE_A_FAIL",
            "experiment_001_ratings_opened": False,
        },
    )


def _gate_a_report(test: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    lines = [
        "## תוצאות L3-SF ושער A",
        "",
        f"ה־post-processing הוקפא לפני test. תוצאת שער A היא **`{test['gate_a']}`**.",
        "",
        "| seed | Precision@4 | Recall@4 | F1@4 | F1@2 | F1@6 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, summary in test["per_seed"].items():
        primary = summary["aggregate"]["4"]
        lines.append(
            f"| {seed} | {_format_float(primary['precision'])} | {_format_float(primary['recall'])} | "
            f"{_format_float(primary['f1'])} | {_format_float(summary['aggregate']['2']['f1'])} | "
            f"{_format_float(summary['aggregate']['6']['f1'])} |"
        )
    inputs = test["gate_a_inputs"]
    lines.extend(
        [
            "",
            f"Median seed F1@4 היה `{_format_float(inputs['median_f1_4px'])}`; precision "
            f"`{_format_float(inputs['median_precision_4px'])}`; recall `{_format_float(inputs['median_recall_4px'])}`. "
            f"ה־baseline הקפוא קיבל F1@4 של `{_format_float(inputs['baseline_f1_4px'])}`, ולכן היתרון המוחלט "
            f"היה `{_format_float(inputs['absolute_f1_advantage'])}`. פער ה־seeds היה "
            f"`{_format_float(inputs['seed_f1_spread'])}`.",
            "",
            "Median F1 לפי run: "
            + ", ".join(f"{run}={_format_float(value)}" for run, value in inputs["run_median_f1_4px"].items())
            + ".",
            "",
            f"פרמטרי baseline: `{baseline['selected_configuration']}`.",
        ]
    )
    return lines


def _write_results(summary: dict[str, Any], test: dict[str, Any], baseline: dict[str, Any], transfer: dict[str, Any]) -> None:
    protocol = _read_json(ARTIFACT_ROOT / "model_protocol.json")
    training = _read_json(ARTIFACT_ROOT / "training_runs.json")
    lines = [
        "# ניסוי 004 — תוצאות pore localization והעברה קפואה ל־SD300",
        "",
        f"**סטטוס:** הושלם ב־{datetime.now(UTC).date().isoformat()} עם הכרעה **`{summary['final_decision']}`**.",
        "",
        f"הפרוטוקול הוקפא ב־commit `{protocol['frozen_configuration']['protocol_commit']}` לפני אימון. "
        "ה־test של L3-SF לא שימש לבחירת architecture, checkpoint, threshold, NMS, loss או preprocessing.",
        "Audit העיוורון נשמר ב־`blindness_audit.json`; לא נפתחו item-level ratings של Experiment 001 לפני freeze של SD300.",
        "",
        "## נתונים ו־ground truth",
        "",
        "החלוקה הקפואה כוללת 88/30/30 קבוצות leakage ו־440/150/150 תמונות train/validation/test. "
        "כל חמשת ה־runs של אותו `pattern + local_index` נשארו יחד. הוסרו בדיוק 241 רשומות coordinate "
        "כפולות; annotations אחרים, לרבות שפתיים ועמומים, נשארו במדד.",
        "",
        f"Ridge period המטרה, שנאמד מ־train בלבד, היה "
        f"`{protocol['frozen_configuration']['ridge_scale']['target_period_px']:.2f} px`.",
        "",
        "## אימון ואירועי runtime",
        "",
        f"אומנו אותם 7,240,225 פרמטרים בשלושת ה־seeds {list(test['per_seed'])}. "
        f"ה־checkpoints נבחרו לפי validation loss בלבד. נשמרו {len(training['attempts'])} ניסיונות בסך הכול; "
        f"{sum(attempt['status'] != 'COMPLETED' for attempt in training['attempts'])} ניסיונות שלא הושלמו נשארו ב־manifest ואינם מוסתרים.",
        "",
    ]
    lines.extend(_gate_a_report(test, baseline))
    lines.extend(["", "## SD300", ""])
    if transfer["status"] == "NOT_RUN_GATE_A_FAIL":
        lines.append("שער A נכשל ולכן SD300 לא נקרא ולא הורץ, בהתאם לתנאי העצירה.")
    else:
        primary = transfer["ppi"]["1000"]
        lines.extend(
            [
                f"ב־1000 ppi היו `{primary['mated_valid_registrations']}` registrations תקפים מתוך 20 mated ו־"
                f"`{primary['non_mated_valid_registrations']}` non-mated. Median Delta הראשי היה "
                f"`{_format_float(primary['primary_median_delta'])}` על `{primary['primary_paired_fingers']}` fingers paired-valid.",
                "",
                f"Bootstrap 95% CI: "
                f"`[{_format_float(primary['primary_bootstrap']['lower'] if primary['primary_bootstrap'] else None)}, "
                f"{_format_float(primary['primary_bootstrap']['upper'] if primary['primary_bootstrap'] else None)}]`.",
                "",
                f"Gate B: **`{summary['gate_b']}`**. Experiment 001 נפתח רק לאחר freeze של 1000 ו־2000 ppi; "
                f"ב־LEVEL3_USABLE נמצאו `{transfer['unblinded_gate_b']['level3_usable_positive_delta_count']}` "
                f"מתוך `{transfer['unblinded_gate_b']['level3_usable_count']}` עם Delta חיובי.",
            ]
        )
        cross = transfer.get("cross_resolution_1000_to_2000")
        if cross:
            lines.extend(
                [
                    "",
                    f"בדיקת sensitivity של אותו impression ב־1000↔2000 נתנה median repeatability "
                    f"`{_format_float(cross['median_repeatability'])}` על `{cross['valid_scores']}` scores תקפים; "
                    "היא אינה נספרת כאוכלוסייה נוספת.",
                ]
            )
    lines.extend(
        [
            "",
            "## הכרעה וגבול הטענה",
            "",
            f"ההכרעה הסופית היא **`{summary['final_decision']}`**.",
            "",
            summary["conclusion"],
            "",
            "אין להסיק מכאן precision/recall אנטומיים על SD300, generalization לכל sensor, שיפור recognition, "
            "matching של זהויות, synthetic-origin detection או liveness. הניסוי נעצר ללא matcher, descriptor או fusion.",
            "",
            "## קבצי ראיה",
            "",
            "המדדים וה־manifests הקומפקטיים נמצאים תחת `artifacts/experiment-004/`. weights, heatmaps, "
            "registrations ו־pixel-bearing derivatives נשארים local תחת `artifacts/experiment-004/local-large/` "
            "ומזוהים באמצעות SHA-256 ב־manifests.",
        ]
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def finalize_localization_fail(test: dict[str, Any], baseline: dict[str, Any]) -> None:
    _write_not_run_sd300()
    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "gate_a": "FAIL",
        "gate_b": "NOT_RUN",
        "final_decision": "LOCALIZATION_FAIL",
        "conclusion": "לא הוכח pore-localization primitive מספיק טוב ב־L3-SF; אין להתקדם ל־matching.",
        "test_metrics_sha256": sha256_file(ARTIFACT_ROOT / "test_metrics.json"),
        "blindness_audit_sha256": sha256_file(ARTIFACT_ROOT / "blindness_audit.json"),
        "sd300_accessed": False,
        "experiment_001_ratings_opened": False,
    }
    write_json(ARTIFACT_ROOT / "summary.json", summary)
    transfer = _read_json(ARTIFACT_ROOT / "sd300_transfer_summary.json")
    _write_results(summary, test, baseline, transfer)


def unblind_and_finalize(test: dict[str, Any], baseline: dict[str, Any]) -> None:
    frozen_marker_path = ARTIFACT_ROOT / "sd300_outputs_frozen.json"
    if not frozen_marker_path.is_file():
        raise RuntimeError("All 1000/2000 SD300 outputs must be frozen before Experiment 001 unblinding")
    transfer_path = ARTIFACT_ROOT / "sd300_transfer_summary.json"
    transfer = _read_json(transfer_path)
    ratings_path = ARTIFACT_ROOT.parent / "experiment-001" / "measurements" / "ratings.json"
    ratings = _read_json(ratings_path)
    if len(ratings) != 20:
        raise RuntimeError("Experiment 001 ratings must contain 20 records")
    classes = {int(row["sample_index"]): _classification(row) for row in ratings}
    class_counts = Counter(classes.values())
    if class_counts["LEVEL3_USABLE"] != 11:
        raise RuntimeError(f"Expected 11 LEVEL3_USABLE fingers, found {class_counts['LEVEL3_USABLE']}")
    finger_rows = transfer["ppi"]["1000"]["finger_median_deltas"]
    deltas = {int(row["sample_index"]): row["median_delta"] for row in finger_rows}
    usable_positive = sum(
        classes[index] == "LEVEL3_USABLE" and deltas.get(index) is not None and deltas[index] > 0
        for index in classes
    )
    by_class: dict[str, dict[str, Any]] = {}
    for classification in sorted(class_counts):
        values = [
            deltas[index]
            for index in classes
            if classes[index] == classification and deltas.get(index) is not None
        ]
        by_class[classification] = {
            "fingers": class_counts[classification],
            "paired_valid_with_delta": len(values),
            "positive_delta_count": sum(value > 0 for value in values),
            "median_delta": float(statistics.median(values)) if values else None,
        }
    blinded = transfer["blinded_gate_b"]
    condition_five = usable_positive >= 8
    all_conditions = all(blinded["conditions_1_to_4"].values()) and condition_five
    if all_conditions:
        gate_b = "TRANSFER_PASS"
    elif blinded["outcome_before_unblind"] == "TRANSFER_FAIL":
        gate_b = "TRANSFER_FAIL"
    else:
        gate_b = "TRANSFER_INCONCLUSIVE"
    transfer.update(
        {
            "status": "UNBLINDED_COMPLETE",
            "experiment_001_ratings_opened": True,
            "unblinded_at_utc": datetime.now(UTC).isoformat(),
            "experiment_001_ratings_sha256": sha256_file(ratings_path),
            "unblinded_gate_b": {
                "level3_usable_count": class_counts["LEVEL3_USABLE"],
                "level3_usable_positive_delta_count": usable_positive,
                "condition_5_at_least_8_of_11": condition_five,
                "classification_counts": dict(class_counts),
                "descriptive_by_class": by_class,
                "gate_b": gate_b,
            },
        }
    )
    write_json(transfer_path, transfer)
    marker = _read_json(frozen_marker_path)
    marker.update(
        {
            "ratings_opened": True,
            "ratings_opened_at_utc": transfer["unblinded_at_utc"],
            "experiment_001_ratings_sha256": transfer["experiment_001_ratings_sha256"],
            "post_unblind_detector_changes": [],
        }
    )
    write_json(frozen_marker_path, marker)
    final_decision = "REAL_TRANSFER_SUPPORTED" if gate_b == "TRANSFER_PASS" else "SYNTHETIC_LOCALIZATION_ONLY"
    conclusion = (
        "מודל pore-localization שנלמד מ־ground truth סינתטי הפיק signal מקומי חוזר ב־SD300 ללא התאמה "
        "ל־SD300, ובמידה גבוהה יותר עבור mated מאשר עבור non-mated תחת הפרוטוקול הקפוא."
        if final_decision == "REAL_TRANSFER_SUPPORTED"
        else "ה־detector הצליח ב־synthetic domain, אך אין evidence מספיק להעברה משכנעת ל־SD300; המשך מחקר "
        "חייב להתמקד ב־domain robustness לפני שימוש ב־pores ל־recognition."
    )
    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "gate_a": test["gate_a"],
        "gate_b": gate_b,
        "final_decision": final_decision,
        "conclusion": conclusion,
        "test_metrics_sha256": sha256_file(ARTIFACT_ROOT / "test_metrics.json"),
        "blindness_audit_sha256": sha256_file(ARTIFACT_ROOT / "blindness_audit.json"),
        "sd300_transfer_summary_sha256": sha256_file(transfer_path),
        "sd300_accessed": True,
        "experiment_001_ratings_opened": True,
    }
    write_json(ARTIFACT_ROOT / "summary.json", summary)
    _write_results(summary, test, baseline, transfer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    test = _read_json(ARTIFACT_ROOT / "test_metrics.json")
    baseline = _read_json(ARTIFACT_ROOT / "baseline_metrics.json")
    if test["gate_a"] == "FAIL":
        finalize_localization_fail(test, baseline)
    else:
        unblind_and_finalize(test, baseline)
    print(_read_json(ARTIFACT_ROOT / "summary.json"))


if __name__ == "__main__":
    main()
