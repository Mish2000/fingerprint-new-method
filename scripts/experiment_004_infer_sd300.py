# ruff: noqa: I001
"""Run the frozen detector on ridge-normalized SD300 caches without OpenCV work."""

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
    from fingerprint_new_method.experiment004_model import infer_heatmap, load_model
    from fingerprint_new_method.experiment004 import (
        ARTIFACT_ROOT,
        EXPERIMENT_ID,
        LOCAL_ROOT,
        TRAINING_SEEDS,
        sha256_bytes,
        sha256_file,
        write_json,
    )

    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppi", required=True, type=int, choices=(1000, 2000))
    arguments = parser.parse_args()
    ppi = arguments.ppi
    model_manifest = _read_json(ARTIFACT_ROOT / "model_manifest.json")
    training = _read_json(ARTIFACT_ROOT / "training_runs.json")
    registration_manifest = _read_json(ARTIFACT_ROOT / f"sd300_registration_manifest_{ppi}.json")
    if registration_manifest["status"] != "REGISTRATIONS_FROZEN_BEFORE_DETECTOR_SCORING":
        raise RuntimeError("Registration statuses must be frozen before detector inference")
    preprocessing = _read_json(ARTIFACT_ROOT / "sd300_preprocessing_manifest.json")
    selected = [
        row
        for key, row in preprocessing["records"].items()
        if key.startswith(f"{ppi}|") and row["status"] == "OK"
    ]
    heatmap_manifest_path = ARTIFACT_ROOT / "sd300_heatmap_manifest.json"
    heatmap_manifest = (
        _read_json(heatmap_manifest_path)
        if heatmap_manifest_path.is_file()
        else {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "configuration_id": model_manifest["configuration_id"],
            "dtype": "float32",
            "records": {},
        }
    )
    for seed in TRAINING_SEEDS:
        run = training["runs"][str(seed)]
        checkpoint_path = PROJECT_ROOT / run["checkpoint_path"]
        if sha256_file(checkpoint_path) != run["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed for seed {seed}")
        model, device, _ = load_model(checkpoint_path)
        for index, row in enumerate(selected, start=1):
            source_id = row["source_id"]
            key = f"{ppi}|{seed}|{source_id}"
            digest = sha256_bytes(source_id.encode())[:16]
            output_path = LOCAL_ROOT / "sd300-heatmaps" / str(ppi) / f"seed-{seed}" / f"{digest}.npy"
            existing = heatmap_manifest["records"].get(key)
            if existing and output_path.is_file() and sha256_file(output_path) == existing["sha256"]:
                continue
            cached = np.load(PROJECT_ROOT / row["relative_local_path"], allow_pickle=False)
            heatmap = infer_heatmap(model, cached.astype(np.float32) / np.float32(255.0), device)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, heatmap.astype(np.float32), allow_pickle=False)
            heatmap_manifest["records"][key] = {
                "ppi": ppi,
                "seed": seed,
                "source_id": source_id,
                "shape": list(heatmap.shape),
                "relative_local_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
            print(f"ppi={ppi} seed={seed} inference={index}/{len(selected)}", flush=True)
        heatmap_manifest["updated_at_utc"] = datetime.now(UTC).isoformat()
        write_json(heatmap_manifest_path, heatmap_manifest)
    print(json.dumps({"status": "SD300_INFERENCE_COMPLETE", "ppi": ppi}, indent=2))


if __name__ == "__main__":
    main()
