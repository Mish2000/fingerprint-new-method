"""Select validation-only operating points, freeze them, and score L3-SF test."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from fingerprint_new_method.experiment004 import (
    ARTIFACT_ROOT,
    EXPERIMENT_ID,
    LOCAL_ROOT,
    PROJECT_ROOT,
    TRAINING_SEEDS,
    aggregate_point_metrics,
    baseline_response,
    canonical_json_sha256,
    deduplicate_points,
    edge_mask,
    extract_peaks,
    gate_a_decision,
    point_metrics,
    read_annotations,
    resolve_source_id,
    sha256_file,
    threshold_cardinality_curve,
    write_csv,
    write_json,
)

THRESHOLDS = tuple(round(value / 100, 2) for value in range(5, 96))
NMS_RADII = (2, 3, 4)
BASELINE_SIGMAS = (1.0, 1.5, 2.0)
BASELINE_PERCENTILES = (98.0, 98.5, 99.0, 99.25, 99.5, 99.75)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _ground_truth(record: dict[str, Any]) -> np.ndarray:
    points, _ = deduplicate_points(read_annotations(resolve_source_id(record["annotation_source_id"])))
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _cache_path(record: dict[str, Any]) -> Path:
    run, stem = str(record["canonical_image_id"]).split("/", maxsplit=1)
    return LOCAL_ROOT / "preprocessed" / "l3sf" / run / f"{stem}.npy"


def _heatmap_path(partition: str, seed: int, record: dict[str, Any]) -> Path:
    return LOCAL_ROOT / "heatmaps" / partition / f"seed-{seed}" / f"{record['canonical_image_id']}.npy"


def _counts_metrics(counts: dict[str, int]) -> dict[str, Any]:
    true_positives = counts["true_positives"]
    predictions = counts["predictions"]
    ground_truth = counts["ground_truth"]
    precision = true_positives / predictions if predictions else (1.0 if ground_truth == 0 else 0.0)
    recall = true_positives / ground_truth if ground_truth else (1.0 if predictions == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **counts,
        "false_positives": predictions - true_positives,
        "false_negatives": ground_truth - true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _sum_counts(rows: Iterable[dict[str, int]]) -> dict[str, int]:
    total = {"predictions": 0, "ground_truth": 0, "true_positives": 0}
    for row in rows:
        for key in total:
            total[key] += int(row[key])
    return total


def select_model_postprocessing(records: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ground_truth = {record["canonical_image_id"]: _ground_truth(record) for record in records}
    counts: dict[tuple[int, int, float], list[dict[str, int]]] = defaultdict(list)
    for seed in TRAINING_SEEDS:
        for index, record in enumerate(records, start=1):
            heatmap = np.load(_heatmap_path("validation", seed, record), allow_pickle=False)
            truth = ground_truth[record["canonical_image_id"]]
            for radius in NMS_RADII:
                peaks = extract_peaks(heatmap, threshold=0.05, nms_radius=radius)
                curve = threshold_cardinality_curve(
                    peaks.coordinates,
                    peaks.scores,
                    truth,
                    THRESHOLDS,
                    tolerance=4.0,
                )
                for threshold in THRESHOLDS:
                    counts[(seed, radius, threshold)].append(curve[threshold])
            if index % 25 == 0:
                print(f"validation model curves seed={seed} images={index}/{len(records)}", flush=True)

    configurations: list[dict[str, Any]] = []
    for radius in NMS_RADII:
        for threshold in THRESHOLDS:
            per_seed = {
                str(seed): _counts_metrics(_sum_counts(counts[(seed, radius, threshold)])) for seed in TRAINING_SEEDS
            }
            f1_values = [per_seed[str(seed)]["f1"] for seed in TRAINING_SEEDS]
            precision_values = [per_seed[str(seed)]["precision"] for seed in TRAINING_SEEDS]
            configurations.append(
                {
                    "threshold": threshold,
                    "nms_radius_px": radius,
                    "per_seed": per_seed,
                    "median_f1": float(statistics.median(f1_values)),
                    "minimum_seed_f1": float(min(f1_values)),
                    "median_precision": float(statistics.median(precision_values)),
                }
            )
    selected = max(
        configurations,
        key=lambda row: (
            row["median_f1"],
            row["minimum_seed_f1"],
            row["median_precision"],
            row["threshold"],
            -row["nms_radius_px"],
        ),
    )
    return selected, configurations


def select_baseline(records: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ground_truth = {record["canonical_image_id"]: _ground_truth(record) for record in records}
    accumulated: dict[tuple[float, float, int], list[dict[str, int]]] = defaultdict(list)
    for index, record in enumerate(records, start=1):
        cached = np.load(_cache_path(record), allow_pickle=False).astype(np.float32) / np.float32(255.0)
        truth = ground_truth[record["canonical_image_id"]]
        for sigma in BASELINE_SIGMAS:
            response = baseline_response(cached, sigma)
            response_thresholds = {percentile: float(np.percentile(response, percentile)) for percentile in BASELINE_PERCENTILES}
            minimum_threshold = min(response_thresholds.values())
            for radius in NMS_RADII:
                peaks = extract_peaks(response, threshold=minimum_threshold, nms_radius=radius)
                curve = threshold_cardinality_curve(
                    peaks.coordinates,
                    peaks.scores,
                    truth,
                    tuple(response_thresholds.values()),
                    tolerance=4.0,
                )
                for percentile, threshold in response_thresholds.items():
                    accumulated[(sigma, percentile, radius)].append(curve[threshold])
        if index % 25 == 0:
            print(f"validation baseline curves images={index}/{len(records)}", flush=True)

    configurations: list[dict[str, Any]] = []
    for sigma in BASELINE_SIGMAS:
        for percentile in BASELINE_PERCENTILES:
            for radius in NMS_RADII:
                metrics = _counts_metrics(_sum_counts(accumulated[(sigma, percentile, radius)]))
                configurations.append(
                    {
                        "sigma": sigma,
                        "percentile": percentile,
                        "nms_radius_px": radius,
                        "metrics": metrics,
                        "false_positives_per_image": metrics["false_positives"] / len(records),
                    }
                )
    selected = max(
        configurations,
        key=lambda row: (
            row["metrics"]["f1"],
            row["metrics"]["precision"],
            -row["false_positives_per_image"],
            row["percentile"],
            -row["sigma"],
            -row["nms_radius_px"],
        ),
    )
    return selected, configurations


def freeze_validation() -> None:
    if (ARTIFACT_ROOT / "test_opened.json").exists():
        raise RuntimeError("Test has already been opened; validation selection cannot be changed")
    split = _read_json(ARTIFACT_ROOT / "split_manifest.json")
    protocol = _read_json(ARTIFACT_ROOT / "model_protocol.json")
    training = _read_json(ARTIFACT_ROOT / "training_runs.json")
    records = [record for record in split["images"] if record["partition"] == "validation"]
    for seed in TRAINING_SEEDS:
        if training["runs"].get(str(seed), {}).get("status") != "COMPLETED":
            raise RuntimeError(f"Seed {seed} is not complete")
        for record in records:
            if not _heatmap_path("validation", seed, record).is_file():
                raise FileNotFoundError(_heatmap_path("validation", seed, record))
    selected_model, model_grid = select_model_postprocessing(records)
    selected_baseline, baseline_grid = select_baseline(records)
    validation_summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "configuration_id": protocol["configuration_id"],
        "partition": "validation",
        "images": len(records),
        "selection_rule": [
            "median seed F1@4px",
            "minimum seed F1",
            "median precision",
            "higher threshold",
            "smaller NMS radius",
        ],
        "selected_model_postprocessing": selected_model,
        "model_grid": model_grid,
        "selected_baseline": selected_baseline,
        "baseline_grid": baseline_grid,
    }
    write_json(ARTIFACT_ROOT / "validation_summary.json", validation_summary)
    baseline_metrics = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "configuration_id": protocol["configuration_id"],
        "selection_partition": "validation",
        "selected_configuration": {
            key: selected_baseline[key] for key in ("sigma", "percentile", "nms_radius_px")
        },
        "validation_metrics_at_4px": selected_baseline["metrics"],
        "test": {"status": "NOT_RUN_BEFORE_PIPELINE_FREEZE"},
    }
    write_json(ARTIFACT_ROOT / "baseline_metrics.json", baseline_metrics)
    weights = []
    for seed in TRAINING_SEEDS:
        run = training["runs"][str(seed)]
        path = PROJECT_ROOT / run["checkpoint_path"]
        weights.append(
            {
                "seed": seed,
                "relative_local_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "best_epoch": run["best_epoch"],
                "best_validation_loss": run["best_validation_loss"],
                "configuration_id": protocol["configuration_id"],
                "regeneration_command": f".conda-env/python scripts/experiment_004_train.py --seed {seed}",
            }
        )
    frozen_inference = {
        "configuration_id": protocol["configuration_id"],
        "preprocessing": protocol["frozen_configuration"]["preprocessing"],
        "ridge_scale": protocol["frozen_configuration"]["ridge_scale"],
        "threshold": selected_model["threshold"],
        "local_maximum_window": [3, 3],
        "nms_radius_px": selected_model["nms_radius_px"],
        "coordinate_convention": "x=column, y=row, direct zero-addressed pixel locations",
        "tiled_inference": protocol["frozen_configuration"]["tiled_inference"],
        "baseline": {key: selected_baseline[key] for key in ("sigma", "percentile", "nms_radius_px")},
    }
    model_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "configuration_id": protocol["configuration_id"],
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "test_pipeline_frozen": True,
        "validation_summary_sha256": sha256_file(ARTIFACT_ROOT / "validation_summary.json"),
        "frozen_inference": frozen_inference,
        "frozen_inference_sha256": canonical_json_sha256(frozen_inference),
        "weights": weights,
        "runtime": training["runtime"],
    }
    write_json(ARTIFACT_ROOT / "model_manifest.json", model_manifest)
    print(
        json.dumps(
            {
                "status": "TEST_PIPELINE_FROZEN",
                "model_threshold": selected_model["threshold"],
                "model_nms_radius_px": selected_model["nms_radius_px"],
                "median_validation_f1": selected_model["median_f1"],
                "baseline": baseline_metrics["selected_configuration"],
                "baseline_validation_f1": selected_baseline["metrics"]["f1"],
            },
            indent=2,
        )
    )


def _evaluate_model_seed(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    threshold: float,
    radius: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        truth = _ground_truth(record)
        heatmap = np.load(_heatmap_path("test", seed, record), allow_pickle=False)
        peaks = extract_peaks(heatmap, threshold=threshold, nms_radius=radius)
        metrics = {str(tolerance): point_metrics(peaks.coordinates, truth, tolerance) for tolerance in (2, 4, 6)}
        primary = metrics["4"]
        edge_truth = edge_mask(truth)
        matched_edge = sum(edge_truth[item["ground_truth_index"]] for item in primary["matches"])
        row = {
            "seed": seed,
            "canonical_image_id": record["canonical_image_id"],
            "run": record["run"],
            "pattern": record["pattern"],
            "predictions": primary["predictions"],
            "ground_truth": primary["ground_truth"],
            "true_positives_4px": primary["true_positives"],
            "false_positives_4px": primary["false_positives"],
            "false_negatives_4px": primary["false_negatives"],
            "precision_4px": primary["precision"],
            "recall_4px": primary["recall"],
            "f1_2px": metrics["2"]["f1"],
            "f1_4px": primary["f1"],
            "f1_6px": metrics["6"]["f1"],
            "mean_localization_error_4px": primary["mean_localization_error"],
            "edge_ground_truth": int(np.count_nonzero(edge_truth)),
            "edge_matched_4px": int(matched_edge),
            "edge_recall_4px": float(matched_edge / np.count_nonzero(edge_truth)) if np.count_nonzero(edge_truth) else None,
            "_metrics": metrics,
        }
        rows.append(row)
        if index % 25 == 0:
            print(f"test model seed={seed} scored={index}/{len(records)}", flush=True)

    def aggregate_subset(subset: Sequence[dict[str, Any]], tolerance: int = 4) -> dict[str, Any]:
        return aggregate_point_metrics([row["_metrics"][str(tolerance)] for row in subset])

    by_run = {run: aggregate_subset([row for row in rows if row["run"] == run]) for run in sorted({r["run"] for r in rows})}
    by_pattern = {
        pattern: aggregate_subset([row for row in rows if row["pattern"] == pattern])
        for pattern in sorted({r["pattern"] for r in rows})
    }
    summary = {
        "seed": seed,
        "aggregate": {str(tolerance): aggregate_subset(rows, tolerance) for tolerance in (2, 4, 6)},
        "by_run_at_4px": by_run,
        "by_pattern_at_4px": by_pattern,
        "edge_at_4px": {
            "ground_truth": sum(row["edge_ground_truth"] for row in rows),
            "matched": sum(row["edge_matched_4px"] for row in rows),
        },
    }
    summary["edge_at_4px"]["recall"] = (
        summary["edge_at_4px"]["matched"] / summary["edge_at_4px"]["ground_truth"]
        if summary["edge_at_4px"]["ground_truth"]
        else None
    )
    return summary, rows


def _evaluate_baseline(records: Sequence[dict[str, Any]], configuration: dict[str, Any]) -> dict[str, Any]:
    rows_by_tolerance: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        image = np.load(_cache_path(record), allow_pickle=False).astype(np.float32) / np.float32(255.0)
        response = baseline_response(image, configuration["sigma"])
        threshold = float(np.percentile(response, configuration["percentile"]))
        peaks = extract_peaks(response, threshold=threshold, nms_radius=configuration["nms_radius_px"])
        truth = _ground_truth(record)
        for tolerance in (2, 4, 6):
            rows_by_tolerance[tolerance].append(point_metrics(peaks.coordinates, truth, tolerance))
    return {str(tolerance): aggregate_point_metrics(rows) for tolerance, rows in rows_by_tolerance.items()}


def score_test() -> None:
    model_manifest = _read_json(ARTIFACT_ROOT / "model_manifest.json")
    if not model_manifest.get("test_pipeline_frozen"):
        raise RuntimeError("Pipeline is not frozen")
    if not (ARTIFACT_ROOT / "test_opened.json").is_file():
        raise RuntimeError("Run frozen test heatmap inference first")
    split = _read_json(ARTIFACT_ROOT / "split_manifest.json")
    records = [record for record in split["images"] if record["partition"] == "test"]
    frozen = model_manifest["frozen_inference"]
    seed_summaries: dict[str, Any] = {}
    per_image_rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        summary, rows = _evaluate_model_seed(
            records,
            seed=seed,
            threshold=frozen["threshold"],
            radius=frozen["nms_radius_px"],
        )
        seed_summaries[str(seed)] = summary
        per_image_rows.extend(rows)
    baseline = _evaluate_baseline(records, frozen["baseline"])
    f1_values = [seed_summaries[str(seed)]["aggregate"]["4"]["f1"] for seed in TRAINING_SEEDS]
    precision_values = [seed_summaries[str(seed)]["aggregate"]["4"]["precision"] for seed in TRAINING_SEEDS]
    recall_values = [seed_summaries[str(seed)]["aggregate"]["4"]["recall"] for seed in TRAINING_SEEDS]
    run_median_f1 = {
        run: float(statistics.median(seed_summaries[str(seed)]["by_run_at_4px"][run]["f1"] for seed in TRAINING_SEEDS))
        for run in ("R1", "R2", "R3", "R4", "R5")
    }
    gate_inputs = {
        "median_f1_4px": float(statistics.median(f1_values)),
        "median_precision_4px": float(statistics.median(precision_values)),
        "median_recall_4px": float(statistics.median(recall_values)),
        "baseline_f1_4px": baseline["4"]["f1"],
        "absolute_f1_advantage": float(statistics.median(f1_values) - baseline["4"]["f1"]),
        "run_median_f1_4px": run_median_f1,
        "seed_f1_spread": float(max(f1_values) - min(f1_values)),
        "all_seeds_above_baseline": all(value > baseline["4"]["f1"] for value in f1_values),
    }
    gate = gate_a_decision(gate_inputs)
    test_metrics = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "configuration_id": model_manifest["configuration_id"],
        "partition": "test",
        "opened_at_utc": _read_json(ARTIFACT_ROOT / "test_opened.json")["opened_at_utc"],
        "frozen_inference_sha256": model_manifest["frozen_inference_sha256"],
        "per_seed": seed_summaries,
        "gate_a_inputs": gate_inputs,
        "gate_a": gate,
    }
    write_json(ARTIFACT_ROOT / "test_metrics.json", test_metrics)
    baseline_metrics = _read_json(ARTIFACT_ROOT / "baseline_metrics.json")
    baseline_metrics["test"] = {"status": "COMPLETED", "metrics": baseline}
    write_json(ARTIFACT_ROOT / "baseline_metrics.json", baseline_metrics)
    csv_rows = [{key: value for key, value in row.items() if key != "_metrics"} for row in per_image_rows]
    write_csv(ARTIFACT_ROOT / "test_per_image.csv", csv_rows, list(csv_rows[0]))
    print(json.dumps({"status": "TEST_SCORED", "gate_a": gate, **gate_inputs}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("validation-freeze", "test-score"))
    arguments = parser.parse_args()
    if arguments.phase == "validation-freeze":
        freeze_validation()
    else:
        score_test()


if __name__ == "__main__":
    main()
