"""Materialize deterministic L3-SF preprocessing without importing PyTorch."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fingerprint_new_method.experiment004 import (
    ARTIFACT_ROOT,
    EXPERIMENT_ID,
    LOCAL_ROOT,
    PROJECT_ROOT,
    preprocess_image,
    resolve_source_id,
    sha256_file,
    write_json,
)


def _cache_path(record: dict[str, Any]) -> Path:
    run, stem = str(record["canonical_image_id"]).split("/", maxsplit=1)
    return LOCAL_ROOT / "preprocessed" / "l3sf" / run / f"{stem}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        action="append",
        choices=("train", "validation", "test"),
        help="Partition to cache; repeat as needed. Default: train and validation only.",
    )
    arguments = parser.parse_args()
    partitions = set(arguments.partition or ("train", "validation"))
    split_path = ARTIFACT_ROOT / "split_manifest.json"
    if not split_path.is_file():
        raise FileNotFoundError("Run experiment_004_prepare.py first")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    manifest_path = ARTIFACT_ROOT / "l3sf_preprocess_cache_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "preprocessing": "robust p01/p99 -> uint8 -> CLAHE(2.0,8x8)",
            "records": {},
        }
    )
    selected = [record for record in split["images"] if record["partition"] in partitions]
    for index, record in enumerate(selected, start=1):
        output_path = _cache_path(record)
        existing = manifest["records"].get(record["canonical_image_id"])
        if existing and output_path.is_file() and sha256_file(output_path) == existing["sha256"]:
            continue
        gray = cv2.imread(str(resolve_source_id(record["image_source_id"])), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not decode {record['image_source_id']}")
        processed_uint8 = np.rint(preprocess_image(gray) * 255.0).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, processed_uint8, allow_pickle=False)
        manifest["records"][record["canonical_image_id"]] = {
            "partition": record["partition"],
            "source_sha256": record["image_sha256"],
            "relative_local_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }
        if index % 50 == 0:
            print(f"cached {index}/{len(selected)}", flush=True)
    manifest["updated_at_utc"] = datetime.now(UTC).isoformat()
    manifest["partition_counts"] = {
        partition: sum(record["partition"] == partition for record in manifest["records"].values())
        for partition in ("train", "validation", "test")
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "CACHE_READY", "partition_counts": manifest["partition_counts"]}, indent=2))


if __name__ == "__main__":
    main()
