# ruff: noqa: I001
"""Train the three preregistered Experiment 004 U-Net seeds."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fingerprint_new_method.paths import PROJECT_ROOT


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required preparation artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _new_training_manifest(
    configuration_id: str,
    experiment_id: str,
    model_configuration: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "configuration_id": configuration_id,
        "model_configuration": model_configuration,
        "runtime": runtime,
        "attempts": [],
        "runs": {},
    }


def main() -> None:
    # See experiment_004_prepare.py: load Torch before OpenCV in a fresh process.
    from fingerprint_new_method.experiment004_model import MODEL_CONFIGURATION, runtime_manifest, train_seed
    from fingerprint_new_method.experiment004 import (
        ARTIFACT_ROOT,
        EXPERIMENT_ID,
        LOCAL_ROOT,
        TRAINING_SEEDS,
        canonical_json_sha256,
        sha256_file,
        write_json,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        choices=TRAINING_SEEDS,
        help="Train one preregistered seed; repeat for multiple. Default: all three.",
    )
    parser.add_argument("--retry-failed", action="store_true", help="Retry a seed whose prior attempt failed")
    arguments = parser.parse_args()

    split = _read_json(ARTIFACT_ROOT / "split_manifest.json")
    protocol = _read_json(ARTIFACT_ROOT / "model_protocol.json")
    if canonical_json_sha256(split) != protocol["split_manifest_sha256"]:
        raise RuntimeError("Split manifest hash differs from the frozen model protocol")
    configuration_id = protocol["configuration_id"]
    training_path = ARTIFACT_ROOT / "training_runs.json"
    training = (
        _read_json(training_path)
        if training_path.is_file()
        else _new_training_manifest(configuration_id, EXPERIMENT_ID, MODEL_CONFIGURATION, runtime_manifest())
    )
    if training["configuration_id"] != configuration_id:
        raise RuntimeError("Existing training runs use a different frozen configuration")
    train_records = [record for record in split["images"] if record["partition"] == "train"]
    validation_records = [record for record in split["images"] if record["partition"] == "validation"]
    if (len(train_records), len(validation_records)) != (440, 150):
        raise RuntimeError(f"Frozen split sizes are wrong: {len(train_records)}, {len(validation_records)}")

    selected_seeds = tuple(arguments.seed or TRAINING_SEEDS)
    for seed in selected_seeds:
        current = training["runs"].get(str(seed))
        if current and current.get("status") == "COMPLETED":
            checkpoint = PROJECT_ROOT / current["checkpoint_path"]
            if checkpoint.is_file() and sha256_file(checkpoint) == current["checkpoint_sha256"]:
                print(f"seed={seed} already completed with matching checkpoint; skipping", flush=True)
                continue
            raise RuntimeError(f"Recorded checkpoint for completed seed {seed} is missing or changed")
        if current and current.get("status") == "FAILED" and not arguments.retry_failed:
            raise RuntimeError(f"Seed {seed} has a recorded failed attempt; pass --retry-failed to preserve and retry it")
        attempt = {
            "seed": seed,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "status": "RUNNING",
        }
        training["attempts"].append(attempt)
        training["runs"][str(seed)] = {"seed": seed, "status": "RUNNING"}
        write_json(training_path, training)
        checkpoint_path = LOCAL_ROOT / "checkpoints" / f"seed-{seed}" / "best.pt"
        try:
            result = train_seed(
                train_records,
                validation_records,
                seed=seed,
                checkpoint_path=checkpoint_path,
            )
            result["checkpoint_sha256"] = sha256_file(checkpoint_path)
            result["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
            result["completed_at_utc"] = datetime.now(UTC).isoformat()
            attempt.update(
                {
                    "status": "COMPLETED",
                    "completed_at_utc": result["completed_at_utc"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                }
            )
            training["runs"][str(seed)] = result
            write_json(training_path, training)
        except BaseException as error:
            failure = {
                "seed": seed,
                "status": "FAILED",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            attempt.update(failure)
            training["runs"][str(seed)] = failure
            write_json(training_path, training)
            raise
    print(json.dumps({"status": "TRAINING_COMMAND_COMPLETE", "runs": training["runs"]}, indent=2))


if __name__ == "__main__":
    main()
