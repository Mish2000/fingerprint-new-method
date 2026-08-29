# ruff: noqa: I001
"""Generate frozen L3-SF heatmaps in a Torch-only execution process."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fingerprint_new_method.paths import PROJECT_ROOT


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    # Torch is intentionally loaded before the conda-forge OpenCV extension.
    from fingerprint_new_method.experiment004_model import (
        infer_heatmap,
        load_model,
        preprocessed_cache_path,
    )
    from fingerprint_new_method.experiment004 import (
        ARTIFACT_ROOT,
        EXPERIMENT_ID,
        LOCAL_ROOT,
        TRAINING_SEEDS,
        sha256_file,
        write_json,
    )

    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", required=True, choices=("validation", "test"))
    arguments = parser.parse_args()

    split = _read_json(ARTIFACT_ROOT / "split_manifest.json")
    protocol = _read_json(ARTIFACT_ROOT / "model_protocol.json")
    training = _read_json(ARTIFACT_ROOT / "training_runs.json")
    configuration_id = protocol["configuration_id"]
    if training["configuration_id"] != configuration_id:
        raise RuntimeError("Training configuration differs from the frozen model protocol")
    if arguments.partition == "test":
        model_manifest = _read_json(ARTIFACT_ROOT / "model_manifest.json")
        if not model_manifest.get("test_pipeline_frozen") or model_manifest["configuration_id"] != configuration_id:
            raise RuntimeError("Test inference requires a matching frozen model_manifest.json")
        marker_path = ARTIFACT_ROOT / "test_opened.json"
        marker = (
            _read_json(marker_path)
            if marker_path.is_file()
            else {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "configuration_id": configuration_id,
                "opened_at_utc": datetime.now(UTC).isoformat(),
                "reason": "Frozen Experiment 004 test heatmap inference",
                "pipeline_freeze_sha256": sha256_file(ARTIFACT_ROOT / "model_manifest.json"),
                "post_test_bug_events": [],
            }
        )
        if marker["configuration_id"] != configuration_id:
            raise RuntimeError("Existing test-open marker has a different configuration")
        write_json(marker_path, marker)

    records = [record for record in split["images"] if record["partition"] == arguments.partition]
    if len(records) != 150:
        raise RuntimeError(f"Expected 150 {arguments.partition} records, found {len(records)}")
    manifest_path = ARTIFACT_ROOT / "l3sf_heatmap_manifest.json"
    manifest = (
        _read_json(manifest_path)
        if manifest_path.is_file()
        else {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "configuration_id": configuration_id,
            "dtype": "float32",
            "records": {},
        }
    )
    if manifest["configuration_id"] != configuration_id:
        raise RuntimeError("Existing heatmaps use another configuration")

    for seed in TRAINING_SEEDS:
        run = training["runs"].get(str(seed))
        if not run or run.get("status") != "COMPLETED":
            raise RuntimeError(f"Seed {seed} did not complete training")
        checkpoint_path = PROJECT_ROOT / run["checkpoint_path"]
        if sha256_file(checkpoint_path) != run["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for seed {seed}")
        model, device, _ = load_model(checkpoint_path)
        for index, record in enumerate(records, start=1):
            key = f"{arguments.partition}|{seed}|{record['canonical_image_id']}"
            output_path = (
                LOCAL_ROOT
                / "heatmaps"
                / arguments.partition
                / f"seed-{seed}"
                / f"{record['canonical_image_id']}.npy"
            )
            existing = manifest["records"].get(key)
            if existing and output_path.is_file() and sha256_file(output_path) == existing["sha256"]:
                continue
            cache_path = preprocessed_cache_path(record)
            cached = np.load(cache_path, allow_pickle=False)
            if cached.dtype != np.uint8 or cached.shape != (512, 512):
                raise ValueError(f"Invalid preprocessing cache {cache_path}")
            heatmap = infer_heatmap(model, cached.astype(np.float32) / np.float32(255.0), device, batch_size=1)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, heatmap.astype(np.float32), allow_pickle=False)
            manifest["records"][key] = {
                "partition": arguments.partition,
                "seed": seed,
                "canonical_image_id": record["canonical_image_id"],
                "relative_local_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
            if index % 25 == 0:
                print(f"partition={arguments.partition} seed={seed} inferred={index}/{len(records)}", flush=True)
        manifest["updated_at_utc"] = datetime.now(UTC).isoformat()
        write_json(manifest_path, manifest)
    print(json.dumps({"status": "INFERENCE_COMPLETE", "partition": arguments.partition}, indent=2))


if __name__ == "__main__":
    main()
