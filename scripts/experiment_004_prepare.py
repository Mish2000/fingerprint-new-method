# ruff: noqa: I001
"""Create the frozen Experiment 004 split, GT summary, and model protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fingerprint_new_method.paths import PROJECT_ROOT

PROTOCOL_PATH = PROJECT_ROOT / "docs" / "experiments" / "004-pore-localization-and-sd300-transfer-protocol.md"


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def verify_protocol_commit() -> str:
    branch = _git("branch", "--show-current")
    if branch != "experiment/004-pore-localization-transfer":
        raise RuntimeError(f"Experiment 004 must run on its dedicated branch, found {branch!r}")
    relative = PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix()
    if _git("status", "--porcelain", "--", relative):
        raise RuntimeError("The Experiment 004 protocol has uncommitted changes")
    commit = _git("log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise RuntimeError("The Experiment 004 protocol has not been committed")
    return commit


def _load_existing(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    # Torch must be imported before conda-forge OpenCV on Windows because both
    # packages carry an OpenMP DLL with the same basename.
    from fingerprint_new_method.experiment004_model import (
        MODEL_CONFIGURATION,
        model_parameter_count,
        runtime_manifest,
    )
    from fingerprint_new_method.experiment004 import (
        ARTIFACT_ROOT,
        BOOTSTRAP_SEED,
        EXPERIMENT_ID,
        SPLIT_SEED,
        TRAINING_SEEDS,
        build_ground_truth_summary,
        build_split_manifest,
        canonical_json_sha256,
        discover_annotated_records,
        sha256_file,
        write_json,
    )
    from fingerprint_new_method.experiment004_transfer import frozen_training_ridge_period

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace matching local preparation artifacts")
    arguments = parser.parse_args()
    protocol_commit = verify_protocol_commit()
    split_path = ARTIFACT_ROOT / "split_manifest.json"
    dedup_path = ARTIFACT_ROOT / "ground_truth_dedup_summary.json"
    model_protocol_path = ARTIFACT_ROOT / "model_protocol.json"
    if not arguments.overwrite and any(path.exists() for path in (split_path, dedup_path, model_protocol_path)):
        raise FileExistsError("Preparation artifacts already exist; use --overwrite only before training")

    records = discover_annotated_records()
    split_manifest = build_split_manifest(records)
    dedup_summary = build_ground_truth_summary(split_manifest)
    train_records = [record for record in split_manifest["images"] if record["partition"] == "train"]
    target_period, ridge_estimates = frozen_training_ridge_period(train_records)
    ridge_summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_partition": "train",
        "target_period_px": target_period,
        "reliable_images": sum(item["status"] == "OK" for item in ridge_estimates),
        "total_images": len(ridge_estimates),
        "per_image": ridge_estimates,
    }
    ridge_summary["content_sha256"] = canonical_json_sha256(ridge_summary)

    frozen_configuration = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_commit": protocol_commit,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "split_seed": SPLIT_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "model": MODEL_CONFIGURATION,
        "preprocessing": {
            "grayscale": True,
            "percentile_clip": [1.0, 99.0],
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": [8, 8],
            "output_range": [0.0, 1.0],
        },
        "augmentation": {
            "rotation_degrees": [-7.0, 7.0],
            "translation_px_each_axis": [-16.0, 16.0],
            "isotropic_scale": [0.95, 1.05],
            "contrast": [0.85, 1.15],
            "brightness": [-0.10, 0.10],
            "border_mode": "REFLECT_101",
        },
        "model_postprocessing_grid": {
            "thresholds": [round(value / 100, 2) for value in range(5, 96)],
            "local_maximum_window": [3, 3],
            "nms_radii_px": [2, 3, 4],
            "selection_metric": "median seed F1@4px on validation",
        },
        "baseline_grid": {
            "gaussian_sigma": [1.0, 1.5, 2.0],
            "response_percentile": [98.0, 98.5, 99.0, 99.25, 99.5, 99.75],
            "nms_radii_px": [2, 3, 4],
        },
        "matching_tolerances_px": [2, 4, 6],
        "primary_tolerance_px": 4,
        "edge_band_px": 8,
        "ridge_scale": {
            "target_period_px": target_period,
            "train_estimates_sha256": ridge_summary["content_sha256"],
            "tile_size": 256,
            "stride": 128,
            "period_search_px": [5, 64],
            "minimum_tiles": 5,
            "maximum_mad_over_median": 0.25,
            "allowed_scale_factor": [0.20, 1.50],
        },
        "tiled_inference": {"tile_size": 512, "overlap": 64, "blend_ramp": 32},
        "registration": {
            "input_blur_sigma": 2.0,
            "sift_nfeatures": 12000,
            "sift_contrast_threshold": 0.02,
            "sift_edge_threshold": 10,
            "sift_sigma": 1.6,
            "mutual_ratio": 0.78,
            "ransac_threshold_px": 6.0,
            "ransac_confidence": 0.999,
            "ransac_max_iterations": 10000,
            "farneback": {
                "pyr_scale": 0.5,
                "levels": 4,
                "winsize": 31,
                "iterations": 5,
                "poly_n": 7,
                "poly_sigma": 1.5,
                "flags": "OPTFLOW_FARNEBACK_GAUSSIAN",
            },
            "flow_consistency_px": 2.0,
        },
        "bootstrap": {"resamples": 10000, "seed": BOOTSTRAP_SEED, "confidence_level": 0.95},
    }
    configuration_id = canonical_json_sha256(frozen_configuration)
    model_protocol = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "configuration_id": configuration_id,
        "frozen_configuration": frozen_configuration,
        "split_manifest_sha256": canonical_json_sha256(split_manifest),
        "ground_truth_summary_sha256": canonical_json_sha256(dedup_summary),
        "parameter_count": model_parameter_count(),
        "runtime_at_preparation": runtime_manifest(),
    }

    write_json(split_path, split_manifest)
    write_json(dedup_path, dedup_summary)
    write_json(ARTIFACT_ROOT / "ridge_period_train_estimates.json", ridge_summary)
    write_json(model_protocol_path, model_protocol)
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "protocol_commit": protocol_commit,
                "configuration_id": configuration_id,
                "groups": split_manifest["group_counts"],
                "images": split_manifest["image_counts"],
                "duplicates_removed": dedup_summary["totals"]["duplicates_removed"],
                "target_ridge_period_px": target_period,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
